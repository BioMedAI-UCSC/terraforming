"""Tests for src.interventions.forcing.

Covers:
  - compute_concentration_ppb: correct molar conversion, zero mass → 0 ppb
  - delta_F_total: monotone with mass, zero when no GHGs
  - update_greenhouse_factor: GHF increases with ΔF, stable at cold T,
    does not change when ΔF ≤ 0, monotone across multiple calls
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.constants import TF_DTYPE
from src.interventions.forcing import (
    compute_concentration_ppb,
    delta_F_total,
    update_greenhouse_factor,
)
from src.interventions.state import GHGState


_CPU = torch.device("cpu")
_MARS_ATM_MASS_KG = 2.5e16    # approximate total Mars atmospheric mass


def _tensor(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=TF_DTYPE, device=_CPU)


def _val(t: torch.Tensor) -> float:
    return float(t.item())


# ---------------------------------------------------------------------------
# compute_concentration_ppb
# ---------------------------------------------------------------------------

class TestComputeConcentrationPpb:

    def test_zero_mass_gives_zero_ppb(self):
        state = GHGState(["CF4"], device=_CPU)
        ppb = compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))
        assert _val(ppb["CF4"]) == pytest.approx(0.0)

    def test_positive_mass_gives_positive_ppb(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e10})
        ppb = compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))
        assert _val(ppb["CF4"]) > 0.0

    def test_ppb_increases_with_injected_mass(self):
        state = GHGState(["SF6"], device=_CPU)
        state.inject({"SF6": 1e9})
        ppb_low = _val(compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))["SF6"])

        state.inject({"SF6": 1e9})
        ppb_high = _val(compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))["SF6"])
        assert ppb_high > ppb_low

    def test_returns_dict_with_all_compounds(self):
        state = GHGState(["CF4", "SF6", "NF3"], device=_CPU)
        ppb = compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))
        assert set(ppb) == {"CF4", "SF6", "NF3"}

    def test_ppb_tensors_on_correct_device(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        ppb = compute_concentration_ppb(state, _tensor(_MARS_ATM_MASS_KG))
        assert ppb["CF4"].device.type == "cpu"

    def test_concentration_formula_is_correct(self):
        """Verify molar mixing ratio formula: (M_i / MW_i) / (M_atm / MW_atm) × 1e9."""
        from src.interventions.compounds import get_compound
        _MARS_ATM_MW = 43.45   # g/mol (matches forcing.py constant)

        mass_kg = 1e10
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": mass_kg})
        atm_mass = _tensor(_MARS_ATM_MASS_KG)

        mw_cf4 = get_compound("CF4").molecular_weight_g_mol
        expected = (mass_kg / mw_cf4) / (_MARS_ATM_MASS_KG / _MARS_ATM_MW) * 1e9

        ppb = compute_concentration_ppb(state, atm_mass)
        assert _val(ppb["CF4"]) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# delta_F_total
# ---------------------------------------------------------------------------

class TestDeltaFTotal:

    def test_zero_ghg_gives_zero_forcing(self):
        state = GHGState(["CF4", "SF6"], device=_CPU)
        dF = delta_F_total(state, _tensor(_MARS_ATM_MASS_KG))
        assert _val(dF) == pytest.approx(0.0)

    def test_positive_injection_gives_positive_forcing(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e12})
        dF = delta_F_total(state, _tensor(_MARS_ATM_MASS_KG))
        assert _val(dF) > 0.0

    def test_forcing_increases_with_more_mass(self):
        state = GHGState(["SF6"], device=_CPU)
        state.inject({"SF6": 1e10})
        dF_low = _val(delta_F_total(state, _tensor(_MARS_ATM_MASS_KG)))

        state.inject({"SF6": 1e10})
        dF_high = _val(delta_F_total(state, _tensor(_MARS_ATM_MASS_KG)))
        assert dF_high > dF_low

    def test_multiple_compounds_sum_correctly(self):
        """ΔF(CF4+SF6) > ΔF(CF4 alone) and > ΔF(SF6 alone)."""
        def _dF_for(*pairs):
            names = [p[0] for p in pairs]
            state = GHGState(names, device=_CPU)
            for name, kg in pairs:
                state.inject({name: kg})
            return _val(delta_F_total(state, _tensor(_MARS_ATM_MASS_KG)))

        dF_both = _dF_for(("CF4", 1e12), ("SF6", 1e12))
        dF_cf4  = _dF_for(("CF4", 1e12))
        dF_sf6  = _dF_for(("SF6", 1e12))
        assert dF_both > dF_cf4
        assert dF_both > dF_sf6

    def test_returns_scalar_tensor(self):
        state = GHGState(["CF4"], device=_CPU)
        dF = delta_F_total(state, _tensor(_MARS_ATM_MASS_KG))
        assert dF.shape == ()


# ---------------------------------------------------------------------------
# update_greenhouse_factor
# ---------------------------------------------------------------------------

class TestUpdateGreenhouseFactor:

    def _mars_with_ghf(self, ghf: float = 1.02) -> Mars:
        return Mars(greenhouse_factor=ghf)

    def test_positive_forcing_increases_ghf(self):
        mars = self._mars_with_ghf()
        baseline_ghf = mars.thermal.greenhouse_factor.clone()
        dF = _tensor(50.0)
        update_greenhouse_factor(mars, dF, baseline_ghf=baseline_ghf)
        new_ghf = _val(mars.thermal.greenhouse_factor)
        assert new_ghf > _val(baseline_ghf)

    def test_zero_forcing_does_not_change_ghf(self):
        """ΔF = 0 → relu gate leaves GHF at baseline, unchanged."""
        mars = self._mars_with_ghf()
        original_ghf = _val(mars.thermal.greenhouse_factor)
        update_greenhouse_factor(
            mars,
            delta_F=_tensor(0.0),
            baseline_ghf=mars.thermal.greenhouse_factor.clone(),
        )
        assert _val(mars.thermal.greenhouse_factor) == pytest.approx(original_ghf)

    def test_negative_forcing_does_not_change_ghf(self):
        """Negative ΔF is gated to zero by relu — GHF stays at baseline."""
        mars = self._mars_with_ghf()
        original_ghf = _val(mars.thermal.greenhouse_factor)
        update_greenhouse_factor(
            mars,
            delta_F=_tensor(-10.0),
            baseline_ghf=mars.thermal.greenhouse_factor.clone(),
        )
        assert _val(mars.thermal.greenhouse_factor) == pytest.approx(original_ghf)

    def test_ghf_always_at_least_one(self):
        """GHF is clamped to minimum 1.0."""
        mars = self._mars_with_ghf(ghf=0.5)   # deliberately below 1
        update_greenhouse_factor(
            mars,
            delta_F=_tensor(0.001),
            baseline_ghf=_tensor(0.5),
        )
        assert _val(mars.thermal.greenhouse_factor) >= 1.0

    def test_ghf_monotone_with_cumulative_forcing(self):
        """Larger cumulative ΔF → larger GHF (monotonicity of formula)."""
        mars_a = self._mars_with_ghf()
        mars_b = self._mars_with_ghf()
        baseline = mars_a.thermal.greenhouse_factor.clone()

        update_greenhouse_factor(mars_a, _tensor(30.0),  baseline_ghf=baseline)
        update_greenhouse_factor(mars_b, _tensor(100.0), baseline_ghf=baseline)

        assert _val(mars_b.thermal.greenhouse_factor) > _val(mars_a.thermal.greenhouse_factor)

    def test_stable_at_cold_temperature(self):
        """Formula must not diverge when surface T ≈ CO₂ frost point (149 K)."""
        mars = Mars(surface_temperature=149.0)
        baseline_ghf = mars.thermal.greenhouse_factor.clone()
        dF = _tensor(50.0)
        update_greenhouse_factor(mars, dF, baseline_ghf=baseline_ghf)
        ghf = _val(mars.thermal.greenhouse_factor)
        assert 1.0 <= ghf < 1e6, f"GHF diverged: {ghf}"

    def test_stable_with_large_forcing(self):
        """Very large ΔF (200 W/m²) should not cause NaN or inf."""
        mars = self._mars_with_ghf()
        baseline_ghf = mars.thermal.greenhouse_factor.clone()
        update_greenhouse_factor(mars, _tensor(200.0), baseline_ghf=baseline_ghf)
        ghf = _val(mars.thermal.greenhouse_factor)
        assert not (ghf != ghf), "GHF is NaN"  # NaN check
        assert ghf < float("inf"), "GHF is inf"

    def test_repeated_calls_use_baseline_not_current(self):
        """Calling with the same baseline_ghf each time avoids compound interest.

        Regression: early implementation multiplied by the *current* GHF on
        each call, causing exponential runaway.
        """
        mars = self._mars_with_ghf(ghf=1.02)
        baseline = mars.thermal.greenhouse_factor.clone()

        # Two years of identical cumulative ΔF should give the same GHF
        update_greenhouse_factor(mars, _tensor(20.0), baseline_ghf=baseline)
        ghf_after_first = _val(mars.thermal.greenhouse_factor)

        # Second call with the same baseline and same ΔF
        update_greenhouse_factor(mars, _tensor(20.0), baseline_ghf=baseline)
        ghf_after_second = _val(mars.thermal.greenhouse_factor)

        assert ghf_after_first == pytest.approx(ghf_after_second, rel=1e-6)

    def test_gradient_flows_from_delta_F_to_ghf(self):
        # Regression B4 (workplan Phase 0): float(delta_F.item()) early-return
        # cut the autograd graph between the forcing and the greenhouse factor.
        mars = self._mars_with_ghf()
        dF = torch.tensor(50.0, dtype=TF_DTYPE, requires_grad=True)
        update_greenhouse_factor(
            mars, dF, baseline_ghf=mars.thermal.greenhouse_factor.clone()
        )
        assert mars.thermal.greenhouse_factor.grad_fn is not None
        mars.thermal.greenhouse_factor.backward()
        assert dF.grad is not None
        assert torch.isfinite(dF.grad)
        assert _val(dF.grad) > 0.0
