"""Tests for src.interventions.controller (InterventionController).

Covers:
  - InterventionController returns correct number of snapshots
  - Snapshot fields are non-negative and physically plausible
  - Temperature increases on average with positive injection
  - ΔF grows each year when injection is continuous
  - GHF is non-decreasing across years
  - Baseline-only run (no injection) leaves GHF and temperature stable
  - Callback is called once per year
  - Unknown compound raises KeyError at construction time
  - Injected GHGs appear in mars.atmosphere.composition (no separate tracker)
  - mars.inject() and mars.decay_ghg() operate on composition directly
  - mars.delta_F reflects current GHG forcing at any point
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.interventions.compounds import COMPOUNDS
from src.interventions.controller import InterventionController, InterventionSnapshot


def _val(t: torch.Tensor) -> float:
    return float(t.item())


def _make_controller(
    injection: dict[str, float] | None = None,
    dt: float = 21600.0,
) -> tuple[InterventionController, Mars]:
    mars = Mars()
    ic = InterventionController(
        mars,
        injection_schedule_kg_yr=injection or {},
        dt=dt,
    )
    return ic, mars


class TestInterventionControllerInit:

    def test_unknown_compound_raises_at_init(self):
        mars = Mars()
        with pytest.raises(KeyError):
            InterventionController(mars, {"FAKE_GAS": 1e9})

    def test_valid_compounds_do_not_raise(self):
        mars = Mars()
        InterventionController(mars, {"CF4": 1e9, "SF6": 5e8})

    def test_composition_unchanged_before_run(self):
        """Injection schedule registered at init must not modify composition yet."""
        mars = Mars()
        keys_before = set(mars.atmosphere.composition.keys())
        InterventionController(mars, {"CF4": 1e9})
        assert set(mars.atmosphere.composition.keys()) == keys_before

    def test_empty_schedule_leaves_composition_unchanged(self):
        mars = Mars()
        keys_before = set(mars.atmosphere.composition.keys())
        InterventionController(mars, {})
        assert set(mars.atmosphere.composition.keys()) == keys_before


class TestMarsUnifiedState:
    """Injected GHGs live in mars.atmosphere.composition — no separate tracker."""

    def test_inject_adds_compound_to_composition(self):
        """CF4 must appear in mars.atmosphere.composition after inject."""
        mars = Mars()
        assert "CF4" not in mars.atmosphere.composition
        mars.inject({"CF4": 1e13})
        assert "CF4" in mars.atmosphere.composition

    def test_inject_increases_partial_pressure(self):
        """Partial pressure of injected compound must be positive after inject."""
        mars = Mars()
        mars.inject({"SF6": 1e13})
        assert _val(mars.atmosphere.composition["SF6"]) > 0.0

    def test_inject_updates_greenhouse_factor(self):
        """GHF must increase immediately after injection."""
        mars = Mars()
        ghf_before = _val(mars.thermal.greenhouse_factor)
        mars.inject({"CF4": 1e13})
        assert _val(mars.thermal.greenhouse_factor) > ghf_before

    def test_delta_F_zero_before_injection(self):
        """delta_F must be zero when no GHGs are in composition."""
        mars = Mars()
        assert _val(mars.delta_F) == pytest.approx(0.0)

    def test_delta_F_positive_after_injection(self):
        mars = Mars()
        mars.inject({"CF4": 1e13})
        assert _val(mars.delta_F) > 0.0

    def test_inject_additive(self):
        """Second inject must stack with first — partial pressure keeps growing."""
        mars = Mars()
        mars.inject({"CF4": 1e13})
        p1 = _val(mars.atmosphere.composition["CF4"])
        mars.inject({"CF4": 1e13})
        p2 = _val(mars.atmosphere.composition["CF4"])
        assert p2 == pytest.approx(p1 * 2, rel=1e-5)

    def test_decay_reduces_partial_pressure(self):
        """decay_ghg() must reduce the partial pressure of injected compounds."""
        mars = Mars()
        mars.inject({"CF4": 1e13})
        p_before = _val(mars.atmosphere.composition["CF4"])
        mars.decay_ghg(dt_years=1.0)
        p_after = _val(mars.atmosphere.composition["CF4"])
        assert p_after < p_before

    def test_decay_does_not_affect_background_gases(self):
        """CO₂, N₂, Ar partial pressures must not change during decay_ghg."""
        mars = Mars()
        mars.inject({"CF4": 1e13})
        co2_before = _val(mars.atmosphere.composition["CO2"])
        mars.decay_ghg(dt_years=1.0)
        assert _val(mars.atmosphere.composition["CO2"]) == pytest.approx(co2_before, rel=1e-8)

    def test_multiple_compounds_all_in_composition(self):
        """All injected compounds must appear in composition after inject."""
        mars = Mars()
        mars.inject({"CF4": 1e9, "SF6": 5e8, "NF3": 2e8})
        for name in ("CF4", "SF6", "NF3"):
            assert name in mars.atmosphere.composition

    def test_mars_state_matches_snapshot_ghf(self):
        """After run(), mars.thermal.greenhouse_factor must equal last snapshot."""
        mars = Mars()
        ic = InterventionController(mars, {"CF4": 1e12}, dt=21600.0)
        history = ic.run(n_years=3)
        assert _val(mars.thermal.greenhouse_factor) == pytest.approx(
            _val(history[-1].greenhouse_factor), rel=1e-6
        )

    def test_mars_delta_F_matches_snapshot(self):
        """mars.delta_F after run() must equal the last snapshot delta_F."""
        mars = Mars()
        ic = InterventionController(mars, {"CF4": 1e12}, dt=21600.0)
        history = ic.run(n_years=3)
        assert _val(mars.delta_F) == pytest.approx(_val(history[-1].delta_F), rel=1e-5)


class TestInterventionControllerRun:

    def test_returns_correct_number_of_snapshots(self):
        ic, _ = _make_controller({"CF4": 1e12})
        assert len(ic.run(n_years=3)) == 3

    def test_snapshots_are_intervention_snapshot_instances(self):
        ic, _ = _make_controller({"CF4": 1e12})
        for snap in ic.run(n_years=2):
            assert isinstance(snap, InterventionSnapshot)

    def test_year_counter_starts_at_one(self):
        ic, _ = _make_controller()
        history = ic.run(n_years=3)
        assert history[0].year == 1
        assert history[-1].year == 3

    def test_surface_temperature_is_positive(self):
        ic, _ = _make_controller({"CF4": 1e12})
        for snap in ic.run(n_years=3):
            assert _val(snap.surface_temperature) > 0

    def test_surface_pressure_is_positive(self):
        ic, _ = _make_controller({"CF4": 1e12})
        for snap in ic.run(n_years=3):
            assert _val(snap.surface_pressure) > 0

    def test_greenhouse_factor_at_least_one(self):
        ic, _ = _make_controller({"CF4": 1e12})
        for snap in ic.run(n_years=3):
            assert _val(snap.greenhouse_factor) >= 1.0

    def test_delta_F_non_negative(self):
        ic, _ = _make_controller({"CF4": 1e12})
        for snap in ic.run(n_years=3):
            assert _val(snap.delta_F) >= 0.0

    def test_delta_F_grows_with_continuous_injection(self):
        ic, _ = _make_controller({"CF4": 1e12})
        history = ic.run(n_years=5)
        assert _val(history[-1].delta_F) > _val(history[0].delta_F)

    def test_ghf_non_decreasing_with_injection(self):
        ic, _ = _make_controller({"CF4": 1e12})
        history = ic.run(n_years=5)
        for i in range(1, len(history)):
            assert _val(history[i].greenhouse_factor) >= _val(
                history[i - 1].greenhouse_factor
            ) - 1e-6

    def test_no_injection_leaves_delta_F_zero(self):
        ic, _ = _make_controller(injection={})
        for snap in ic.run(n_years=3):
            assert _val(snap.delta_F) == pytest.approx(0.0)

    def test_ghg_partial_pressure_has_injected_compound(self):
        ic, _ = _make_controller({"SF6": 1e11})
        for snap in ic.run(n_years=2):
            assert "SF6" in snap.ghg_partial_pressure_Pa

    def test_cumulative_injected_increases_each_year(self):
        ic, _ = _make_controller({"CF4": 1e12})
        history = ic.run(n_years=3)
        cum0 = _val(history[0].cumulative_injected_kg["CF4"])
        cum2 = _val(history[2].cumulative_injected_kg["CF4"])
        assert cum2 > cum0

    def test_time_s_increases_each_year(self):
        ic, _ = _make_controller({"CF4": 1e12})
        history = ic.run(n_years=3)
        for i in range(1, len(history)):
            assert _val(history[i].time_s) > _val(history[i - 1].time_s)

    def test_callback_called_once_per_year(self):
        call_count = [0]
        ic, _ = _make_controller({"CF4": 1e12})
        ic.run(n_years=4, callback=lambda s: call_count.__setitem__(0, call_count[0] + 1))
        assert call_count[0] == 4

    def test_callback_receives_snapshot(self):
        received = []
        ic, _ = _make_controller({"CF4": 1e12})
        ic.run(n_years=2, callback=received.append)
        assert all(isinstance(s, InterventionSnapshot) for s in received)

class TestDifferentiableSchedule:
    """Regression B2 (workplan Phase 0): the controller coerced the schedule
    to Python floats at __init__, so gradients could never reach it."""

    def test_final_temperature_has_grad_fn_given_tensor_schedule(self):
        kg = torch.tensor(1e12, requires_grad=True)
        mars = Mars(device="cpu")
        ic = InterventionController(mars, {"CF4": kg}, dt=21600.0)
        history = ic.run(n_years=1)
        assert history[-1].surface_temperature.grad_fn is not None
        assert history[-1].greenhouse_factor.grad_fn is not None

    def test_gradient_of_temperature_wrt_schedule_is_finite_and_positive(self):
        """d(annual-mean T)/d(kg injected) must be finite and > 0 — more GHG warms."""
        kg = torch.tensor(1e12, requires_grad=True)
        mars = Mars(device="cpu")
        ic = InterventionController(mars, {"CF4": kg}, dt=21600.0)
        history = ic.run(n_years=1)
        history[-1].surface_temperature.backward()
        assert kg.grad is not None
        assert torch.isfinite(kg.grad)
        assert float(kg.grad.item()) > 0.0

    def test_float_schedule_matches_tensor_schedule(self):
        """Plain-float schedules (the CLI path) must produce identical results."""
        mars_f = Mars(device="cpu")
        mars_t = Mars(device="cpu")
        hist_f = InterventionController(mars_f, {"CF4": 1e12}, dt=43200.0).run(n_years=1)
        # TF_DTYPE (float64) — a float32 tensor cannot represent 1e12 exactly
        hist_t = InterventionController(
            mars_t, {"CF4": torch.tensor(1e12, dtype=torch.float64)}, dt=43200.0
        ).run(n_years=1)
        assert _val(hist_f[-1].surface_temperature) == pytest.approx(
            _val(hist_t[-1].surface_temperature), rel=1e-12
        )
        assert _val(hist_f[-1].greenhouse_factor) == pytest.approx(
            _val(hist_t[-1].greenhouse_factor), rel=1e-12
        )


class TestInterventionControllerLong:

    @pytest.mark.slow
    def test_50_year_run_temperature_rises(self):
        mars = Mars()
        ic = InterventionController(mars, {"CF4": 1e9}, dt=21600.0)
        history = ic.run(n_years=50)
        T_start = _val(history[0].surface_temperature)
        T_end   = _val(history[-1].surface_temperature)
        assert T_end > T_start + 5.0, (
            f"Expected temperature rise > 5 K, got {T_end - T_start:.1f} K"
        )
