"""Tests for src.engine.diagnostics (mass-conservation harness).

Covers:
  - atmosphere_mass_kg: hydrostatic conversion P -> kg
  - mass_budget_residual_kg: closes for FAST and RK4 runs, accounts for
    injection, and trips on a deliberately broken (mass-leaking) state
  - composition_consistency_residual_pa: zero after runs in both modes

Regression context: before this harness existed, the thermal tide lived
inside the prognostic pressure and the budget could not close (audit finding
#4, -0.139 Pa-equivalent over 10 sols).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.celestials import Mars
from src.constants import TF_DTYPE
from src.engine.diagnostics import (
    atmosphere_mass_kg,
    composition_consistency_residual_pa,
    mass_budget_residual_kg,
)
from src.engine.time_controller import Accuracy, TimeController


_SOL = 88_775.244
# 1e9 kg ≈ 2.6e-5 Pa — generous vs float64 rounding, tiny vs any real leak
# (the old tide leak was ~1e15 kg within a single sol).
_TOL_KG = 1.0e9


def _capture(mars: Mars):
    return (
        mars.atmosphere.surface_pressure.clone(),
        mars.water.ice_mass.clone(),
    )


class TestAtmosphereMassKg:

    def test_matches_hand_computed_hydrostatic_mass(self):
        mars = Mars(device="cpu")
        g = float(mars.intrinsic_params.gravity)
        r = float(mars.intrinsic_params.radius)
        P = float(mars.atmosphere.surface_pressure)
        expected = P * 4.0 * math.pi * r**2 / g
        got = atmosphere_mass_kg(
            mars.atmosphere.surface_pressure,
            mars.intrinsic_params.gravity,
            mars.intrinsic_params.radius,
        )
        assert float(got) == pytest.approx(expected, rel=1e-12)


class TestMassBudgetCloses:

    def test_fast_mode_budget_closes_over_three_sols(self):
        mars = Mars(device="cpu")
        p0, ice0 = _capture(mars)
        dur = 3 * _SOL
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=dur)
        res = mass_budget_residual_kg(mars, p0, ice0, dur)
        assert abs(float(res)) < _TOL_KG

    def test_rk4_mode_budget_closes_over_two_sols(self):
        mars = Mars(device="cpu")
        p0, ice0 = _capture(mars)
        dur = 2 * _SOL
        TimeController(mars, dt=300.0, accuracy=Accuracy.ACCURATE).run(duration=dur)
        res = mass_budget_residual_kg(mars, p0, ice0, dur)
        assert abs(float(res)) < _TOL_KG

    def test_budget_accounts_for_injected_mass(self):
        mars = Mars(device="cpu")
        p0, ice0 = _capture(mars)
        kg = 1.0e12
        mars.inject({"SF6": kg})
        res = mass_budget_residual_kg(mars, p0, ice0, elapsed_s=0.0, injected_kg=kg)
        assert abs(float(res)) < _TOL_KG

    def test_residual_trips_on_deliberate_mass_leak(self):
        """A term that adds pressure from nowhere must be caught."""
        mars = Mars(device="cpu")
        p0, ice0 = _capture(mars)
        mars.atmosphere.surface_pressure = mars.atmosphere.surface_pressure + 10.0
        res = mass_budget_residual_kg(mars, p0, ice0, elapsed_s=0.0)
        assert abs(float(res)) > 1.0e14  # 10 Pa ≈ 3.9e14 kg


class TestCompositionConsistency:

    def test_zero_at_init(self):
        assert float(composition_consistency_residual_pa(Mars(device="cpu"))) == (
            pytest.approx(0.0, abs=1e-9)
        )

    def test_zero_after_fast_run(self):
        mars = Mars(device="cpu")
        TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=2 * _SOL)
        assert float(composition_consistency_residual_pa(mars)) < 1e-9

    def test_zero_after_rk4_run(self):
        mars = Mars(device="cpu")
        TimeController(mars, dt=300.0, accuracy=Accuracy.ACCURATE).run(duration=_SOL)
        assert float(composition_consistency_residual_pa(mars)) < 1e-9

    def test_zero_after_inject_and_decay(self):
        mars = Mars(device="cpu")
        mars.inject({"CF4": 1e13, "SF6": 5e12})
        assert float(composition_consistency_residual_pa(mars)) < 1e-9
        mars.decay_ghg(dt_years=5.0)
        assert float(composition_consistency_residual_pa(mars)) < 1e-9
