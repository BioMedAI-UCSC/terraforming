"""Regression tests for the pressure-handling fixes in mars.py.

Each class targets one verified bug from the pressure audit:
  - Injected GHG mass never reached total pressure (partial +2.58 Pa,
    total +0.0000 Pa)
  - Composition partial pressures were static while total P evolved
    (CO2 stuck at 580 Pa forever)
  - Init inconsistency: P0=610 vs sum(composition)=608.2; atmospheric_mass
    hardcoded 2.5e16 kg vs hydrostatic 2.367e16
  - The +/-30 Pa local thermal tide lived inside the global mass variable
"""

from __future__ import annotations

import math

import pytest
import torch

from src.celestials import Mars
from src.constants import TF_DTYPE
from src.engine.time_controller import Accuracy, TimeController


_SOL = 88_775.244


def _val(t: torch.Tensor) -> float:
    return float(t.item())


def _comp_sum(mars: Mars) -> float:
    return float(sum(v for v in mars.atmosphere.composition.values()))


class TestInitConsistency:

    def test_composition_sums_to_surface_pressure_default(self):
        # Regression: default init had P0=610 Pa vs sum(composition)=608.2 Pa.
        mars = Mars(device="cpu")
        assert _comp_sum(mars) == pytest.approx(_val(mars.atmosphere.surface_pressure), rel=1e-12)

    def test_composition_sums_to_surface_pressure_custom(self):
        mars = Mars(device="cpu", surface_pressure=705.0)
        assert _comp_sum(mars) == pytest.approx(705.0, rel=1e-12)

    def test_composition_scales_with_elevation_correction(self):
        mars = Mars(device="cpu", surface_pressure=610.0, elevation_m=-4000.0)
        P = _val(mars.atmosphere.surface_pressure)
        assert P == pytest.approx(610.0 * math.exp(4000.0 / 11_100.0), rel=1e-9)
        assert _comp_sum(mars) == pytest.approx(P, rel=1e-12)

    def test_atmospheric_mass_is_hydrostatic(self):
        # Regression: hardcoded 2.5e16 kg was ~5.6 % above P·A/g.
        mars = Mars(device="cpu")
        g = _val(mars.intrinsic_params.gravity)
        A = 4.0 * math.pi * _val(mars.intrinsic_params.radius) ** 2
        expected = _val(mars.atmosphere.surface_pressure) * A / g
        assert _val(mars.atmosphere.atmospheric_mass) == pytest.approx(expected, rel=1e-12)


class TestInjectionMassBudget:

    def test_inject_raises_total_pressure_hydrostatically(self):
        # Regression: injecting 1e14 kg CF4 added 2.577 Pa of partial pressure
        # but changed total surface_pressure by exactly 0.
        mars = Mars(device="cpu")
        g = _val(mars.intrinsic_params.gravity)
        A = 4.0 * math.pi * _val(mars.intrinsic_params.radius) ** 2
        kg = 1.0e14
        P_before = _val(mars.atmosphere.surface_pressure)
        mars.inject({"CF4": kg})
        assert _val(mars.atmosphere.surface_pressure) - P_before == pytest.approx(
            kg * g / A, rel=1e-9
        )

    def test_decay_lowers_total_pressure_by_decayed_partials(self):
        mars = Mars(device="cpu")
        mars.inject({"CF4": 1e14})
        P_before = _val(mars.atmosphere.surface_pressure)
        pp_before = _val(mars.atmosphere.composition["CF4"])
        mars.decay_ghg(dt_years=10.0)
        pp_lost = pp_before - _val(mars.atmosphere.composition["CF4"])
        assert pp_lost > 0.0
        assert P_before - _val(mars.atmosphere.surface_pressure) == pytest.approx(
            pp_lost, rel=1e-9
        )

    def test_composition_stays_consistent_through_inject_decay(self):
        mars = Mars(device="cpu")
        mars.inject({"SF6": 5e13})
        mars.decay_ghg(dt_years=3.0)
        assert _comp_sum(mars) == pytest.approx(
            _val(mars.atmosphere.surface_pressure), rel=1e-12
        )


class TestCompositionTracksEvolvingPressure:

    def test_co2_partial_pressure_follows_cap_exchange_fast(self):
        # Regression: composition["CO2"] stayed at its init value forever
        # while surface_pressure moved tens of Pa with the caps.
        mars = Mars(device="cpu")
        co2_0 = _val(mars.atmosphere.composition["CO2"])
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=5 * _SOL)
        dP = _val(mars.atmosphere.surface_pressure) - 610.0
        assert abs(dP) > 1.0  # the caps actually moved pressure
        assert _val(mars.atmosphere.composition["CO2"]) - co2_0 == pytest.approx(
            dP, abs=1e-6
        )
        assert _comp_sum(mars) == pytest.approx(
            _val(mars.atmosphere.surface_pressure), rel=1e-12
        )

    def test_co2_partial_pressure_follows_cap_exchange_rk4(self):
        mars = Mars(device="cpu")
        TimeController(mars, dt=300.0, accuracy=Accuracy.ACCURATE).run(duration=_SOL)
        assert _comp_sum(mars) == pytest.approx(
            _val(mars.atmosphere.surface_pressure), rel=1e-12
        )


class TestTideIsDiagnosticOverlay:

    def _no_cap_mars(self) -> Mars:
        mars = Mars(device="cpu")
        z = torch.zeros((), dtype=TF_DTYPE)
        mars.water.ice_mass_north = z.clone()
        mars.water.ice_mass_south = z.clone()
        mars.water.ice_mass = z.clone()
        return mars

    def test_prognostic_pressure_is_tide_free(self):
        # Regression: the ±30 Pa local tide oscillated the global mass
        # variable (+28.9 Pa half a sol in, with caps started at zero).
        # Now the only pressure change is genuine mass exchange: the dark
        # pole condenses ~0.08 Pa onto the (empty) cap — accounted exactly
        # by the ice ledger — and the tide contributes nothing.
        mars = self._no_cap_mars()
        P0 = _val(mars.atmosphere.surface_pressure)
        g = _val(mars.intrinsic_params.gravity)
        A = 4.0 * math.pi * _val(mars.intrinsic_params.radius) ** 2
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=0.5 * _SOL)
        dP = _val(mars.atmosphere.surface_pressure) - P0
        d_ice = _val(mars.water.ice_mass)  # started at zero
        assert abs(dP) < 0.5, "tide-scale signal leaked into the mass budget"
        assert dP == pytest.approx(-d_ice * g / A, abs=1e-6)

    def test_observed_pressure_carries_the_tide(self):
        mars = self._no_cap_mars()
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=0.5 * _SOL)
        obs = _val(mars.observed_surface_pressure())
        mean = _val(mars.atmosphere.surface_pressure)
        assert abs(obs - mean) > 5.0  # tide visible (amplitude 30 Pa)
        assert abs(obs - mean) <= 2 * 30.0 + 1e-6

    def test_overlay_matches_closed_form(self):
        mars = self._no_cap_mars()
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=0.3 * _SOL)
        omega = 2.0 * math.pi / _val(mars.intrinsic_params.rotation_period)
        t = _val(mars.elapsed_time)
        phase = float(mars._TIDE_PHASE)
        expected = 30.0 * (math.cos(omega * t + phase) - math.cos(phase))
        got = _val(mars.observed_surface_pressure()) - _val(mars.atmosphere.surface_pressure)
        assert got == pytest.approx(expected, abs=1e-9)

    def test_observed_equals_prognostic_at_t_zero(self):
        mars = Mars(device="cpu")
        assert _val(mars.observed_surface_pressure()) == pytest.approx(
            _val(mars.atmosphere.surface_pressure), abs=1e-12
        )

    def test_snapshot_records_observed_pressure(self):
        mars = self._no_cap_mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        history = tc.run(duration=1 * _SOL)
        press = [(_val(s.time), _val(s.surface_pressure)) for s in history]
        amplitudes = [p - press[0][1] for _, p in press]
        # The diurnal wave must still be visible in outputs (CLI plots/CSV)
        assert max(amplitudes) - min(amplitudes) > 20.0
