"""Tests for the smooth ice-exhaustion gates in mars.py (Mars(smooth_gates=...)).

Covers the P1 discontinuity bug from the differentiability audit
(docs/ideas/iclr2027-workplan.md, item D1): the hard gate
``torch.where((ice <= 0) & (dM < 0), 0, dM)`` is correct physics but has
zero gradient once a polar cap empties, so gradient-based optimisation is
blind exactly in the CO2-collapse regime.

  - Hard gate stays the default and its semantics are unchanged
  - Smooth and hard modes agree while ice >> ice_ref (sigmoid saturates)
  - Gradient counterexample: d(flux)/d(ice) is zero with the hard gate and
    finite/nonzero with the smooth gate at an exhausted cap
  - End-to-end: d(P_final)/d(initial ice) flows through a fast-mode rollout
    only in smooth mode
  - Physical invariants hold in smooth mode (ice >= 0, T > 0)
  - ice_ref_kg validation
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.constants import TF_DTYPE
from src.engine.time_controller import Accuracy, TimeController


_SOL = 88_775.244  # s


def _val(t: torch.Tensor) -> float:
    return float(t.item())


def _northern_summer_mars(**kwargs) -> Mars:
    """Mars at Ls=90 (northern summer solstice) — the north cap sublimates."""
    return Mars(initial_ls_deg=90.0, device="cpu", **kwargs)


def _sunlit(mars: Mars) -> Mars:
    """Give the planet a realistic solar flux (it is 0 until advance_orbit)."""
    mars.radiation.solar_flux = torch.tensor(590.0, dtype=TF_DTYPE)
    return mars


def _pack_state(mars: Mars) -> torch.Tensor:
    # State is [T, P, M_north, M_south] — the two caps are independent.
    return torch.stack(
        [
            mars.thermal.surface_temperature,
            mars.atmosphere.surface_pressure,
            mars.water.ice_mass_north,
            mars.water.ice_mass_south,
        ]
    )


class TestConstructor:

    def test_smooth_gates_defaults_to_false(self):
        assert Mars(device="cpu")._smooth_gates is False

    def test_ice_ref_raises_on_nonpositive(self):
        with pytest.raises(ValueError):
            Mars(device="cpu", ice_ref_kg=0.0)
        with pytest.raises(ValueError):
            Mars(device="cpu", ice_ref_kg=-1e12)


class TestGateSemantics:

    def test_empty_caps_never_lose_mass(self):
        """An empty cap must not sublimate in either mode (hard gate zeroes
        the flux; tanh(0) = 0 fades it to exactly zero).  At Ls=90 the sunlit
        north cap would sublimate and the dark south cap condenses, so with
        both caps empty the net flux must be the condensation term alone
        (>= 0) — mass cannot leave a cap that has none."""
        for smooth in (False, True):
            mars = _sunlit(_northern_summer_mars(smooth_gates=smooth))
            mars.water.ice_mass_north = torch.zeros((), dtype=TF_DTYPE)
            mars.water.ice_mass_south = torch.zeros((), dtype=TF_DTYPE)
            derivs = mars.compute_derivatives(_pack_state(mars))
            # derivs = [dT, dP, dM_north, dM_south]; neither empty cap may lose
            # mass (sublimation gated to 0), so both cap tendencies are >= 0.
            assert _val(derivs[2]) >= 0.0, f"north smooth={smooth}"
            assert _val(derivs[3]) >= 0.0, f"south smooth={smooth}"

    def test_condensation_is_never_gated(self):
        """dM > 0 (condensing) must pass through even at ice = 0, both modes."""
        for smooth in (False, True):
            mars = Mars(device="cpu", smooth_gates=smooth)  # Ls=251, dark poles
            mars.water.ice_mass_north = torch.zeros((), dtype=TF_DTYPE)
            mars.water.ice_mass_south = torch.zeros((), dtype=TF_DTYPE)
            # No sunlight: Q_in = 0 < Q_out_pole → net condensation (dM > 0)
            derivs = mars.compute_derivatives(_pack_state(mars))
            assert _val(derivs[2]) > 0.0, f"smooth={smooth}"


class TestSmoothHardAgreement:

    def test_derivatives_identical_when_ice_abundant(self):
        """With both caps >> ice_ref (1e12), tanh saturates to exactly 1.0 in
        float64, so the two modes must agree bit-for-bit.  (Caps are set
        explicitly because at Ls=90 the north cap starts empty by
        construction.)"""
        abundant = torch.tensor(2.5e15, dtype=TF_DTYPE)
        planets = {}
        for smooth in (False, True):
            mars = _sunlit(_northern_summer_mars(smooth_gates=smooth))
            mars.water.ice_mass_north = abundant.clone()
            mars.water.ice_mass_south = abundant.clone()
            mars.water.ice_mass = mars.water.ice_mass_north + mars.water.ice_mass_south
            planets[smooth] = mars
        d_hard   = planets[False].compute_derivatives(_pack_state(planets[False]))
        d_smooth = planets[True].compute_derivatives(_pack_state(planets[True]))
        assert torch.equal(d_hard, d_smooth)

    def test_fast_rollout_agreement_when_ice_abundant(self):
        """2-sol FAST rollouts must agree to <0.5% while ice >> ice_ref."""
        results = {}
        for smooth in (False, True):
            mars = _northern_summer_mars(smooth_gates=smooth)
            tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
            history = tc.run(duration=2 * _SOL)
            results[smooth] = history[-1]
        for attr in ("surface_temperature", "surface_pressure", "ice_mass"):
            v_hard   = _val(getattr(results[False], attr))
            v_smooth = _val(getattr(results[True], attr))
            assert v_smooth == pytest.approx(v_hard, rel=5e-3), attr


class TestGradientThroughExhaustion:
    """The bug being fixed: the hard gate's gradient is zero at an empty cap."""

    def _flux_grad_at_empty_cap(self, smooth: bool) -> torch.Tensor | None:
        # The north cap (index 2) is the sunlit, empty, sublimating one. Put it in
        # the state with requires_grad and differentiate its own tendency w.r.t.
        # it — the autodiff path the RK4 rollout uses (caps read from y).
        mars = _sunlit(_northern_summer_mars(smooth_gates=smooth))
        ice_n = torch.zeros((), dtype=TF_DTYPE, requires_grad=True)
        y = torch.stack([
            mars.thermal.surface_temperature.detach(),
            mars.atmosphere.surface_pressure.detach(),
            ice_n,
            mars.water.ice_mass_south.detach(),
        ])
        out = mars.compute_derivatives(y)[2]  # north-cap tendency dM_north
        if out.grad_fn is None:
            return None  # output is disconnected from ice_n entirely
        (grad,) = torch.autograd.grad(out, ice_n, allow_unused=True)
        return grad

    def test_hard_gate_gradient_is_dead_at_empty_cap(self):
        """Documents the counterexample: hard mode gives no gradient signal."""
        grad = self._flux_grad_at_empty_cap(smooth=False)
        assert grad is None or _val(grad) == pytest.approx(0.0)

    def test_smooth_gate_gradient_is_alive_at_empty_cap(self):
        grad = self._flux_grad_at_empty_cap(smooth=True)
        assert grad is not None
        assert torch.isfinite(grad)
        assert _val(grad) != 0.0

    def _pressure_grad_after_rollout(self, smooth: bool) -> torch.Tensor | None:
        """d(P_final)/d(initial cap ice) through a 3-sol FAST rollout in which
        the sublimation gate is active (ice ~ ice_ref)."""
        mars = _northern_summer_mars(smooth_gates=smooth, ice_mass=1e12)
        ice0 = torch.tensor(5e11, dtype=TF_DTYPE, requires_grad=True)
        mars.water.ice_mass_north = ice0
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        tc.run(duration=3 * _SOL)
        P = mars.atmosphere.surface_pressure
        if P.grad_fn is None:
            return None  # pressure is disconnected from ice0 entirely
        (grad,) = torch.autograd.grad(P, ice0, allow_unused=True)
        return grad

    def test_pressure_insensitive_to_ice_with_hard_gate(self):
        # Regression counterexample: with the hard gate the sublimation flux
        # never depends on the ice amount, so the optimizer sees d(P)/d(ice)=0.
        grad = self._pressure_grad_after_rollout(smooth=False)
        assert grad is None or _val(grad) == pytest.approx(0.0)

    def test_pressure_sensitive_to_ice_with_smooth_gate(self):
        grad = self._pressure_grad_after_rollout(smooth=True)
        assert grad is not None
        assert torch.isfinite(grad)
        assert _val(grad) != 0.0


class TestSmoothModeInvariants:

    def test_ice_stays_nonnegative_through_exhaustion(self):
        """Run a small cap to exhaustion in smooth mode: ice >= 0, T > 0."""
        mars = _northern_summer_mars(smooth_gates=True, ice_mass=1e10)
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        history = tc.run(duration=5 * _SOL)
        for snap in history:
            assert _val(snap.ice_mass) >= 0.0
            assert _val(snap.surface_temperature) > 0.0
