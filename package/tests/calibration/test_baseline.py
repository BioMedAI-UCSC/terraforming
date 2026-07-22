"""Tests for src.calibration.baseline.

Two layers:
  - Fast: summarize_baseline / BaselineStability on synthetic UnforcedRun arrays
    (a hand-built perfect limit cycle, and a drifting one) — no model run.
  - Slow (`@pytest.mark.slow`): the real unforced model reproduces present-day
    Mars (~6 mb, seasonal swing) as a stable, non-drifting state.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.baseline import (
    BaselineStability,
    UnforcedRun,
    assess_baseline,
    run_unforced,
    summarize_baseline,
)

YEAR_S = 5.93568e7


def _synthetic_run(n_years: int, mean_pa: float, swing_pa: float, drift_pa_per_year: float):
    """A clean multi-year series: repeating cosine cycle + optional linear drift."""
    per_year = 400
    t = np.linspace(0, n_years * YEAR_S, n_years * per_year, endpoint=False)
    phase = 2 * np.pi * (t / YEAR_S)
    yr = t / YEAR_S
    P = mean_pa + 0.5 * swing_pa * np.cos(phase) + drift_pa_per_year * yr
    T = 210.0 + 5.0 * np.cos(phase)
    ice = 1.5e15 + 1e14 * np.cos(phase)
    return UnforcedRun(time_s=t, pressure_pa=P, temperature_k=T, ice_kg=ice, year_s=YEAR_S)


class TestSummarizeStableCycle:

    def test_recovers_mean_and_swing(self):
        run = _synthetic_run(4, mean_pa=610.0, swing_pa=100.0, drift_pa_per_year=0.0)
        s = summarize_baseline(run, spinup_years=1)
        assert s.mean_pressure_pa == pytest.approx(610.0, abs=1.0)
        assert s.mean_pressure_mb == pytest.approx(6.1, abs=0.02)
        assert s.seasonal_swing_pa == pytest.approx(100.0, rel=0.05)
        assert s.seasonal_swing_pct == pytest.approx(100.0 * 100.0 / 610.0, rel=0.05)

    def test_perfect_cycle_has_zero_drift_and_repeatable_swing(self):
        run = _synthetic_run(5, mean_pa=610.0, swing_pa=100.0, drift_pa_per_year=0.0)
        s = summarize_baseline(run, spinup_years=1)
        assert abs(s.pressure_drift_pa_per_year) < 1e-6
        assert s.swing_repeatability_pa < 1e-6
        assert s.is_stable()

    def test_detects_drift_as_unstable(self):
        run = _synthetic_run(5, mean_pa=610.0, swing_pa=100.0, drift_pa_per_year=5.0)
        s = summarize_baseline(run, spinup_years=1)
        assert s.pressure_drift_pa_per_year == pytest.approx(5.0, rel=1e-3)
        assert not s.is_stable(drift_tol_pa_per_year=1.0)

    def test_pct_per_century_scaling(self):
        run = _synthetic_run(5, mean_pa=600.0, swing_pa=50.0, drift_pa_per_year=0.6)
        s = summarize_baseline(run, spinup_years=1)
        # 0.6 Pa/yr * 100 yr / 600 Pa * 100% = 10 %/century
        assert s.pressure_drift_pct_per_century == pytest.approx(10.0, rel=1e-2)

    def test_rejects_run_shorter_than_spinup(self):
        run = _synthetic_run(1, mean_pa=610.0, swing_pa=100.0, drift_pa_per_year=0.0)
        with pytest.raises(ValueError):
            summarize_baseline(run, spinup_years=1)


class TestAssessBaselineValidation:

    def test_rejects_nonpositive_span(self):
        with pytest.raises(ValueError):
            assess_baseline(n_years=1, spinup_years=1)


# ── Slow: the real unforced model ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline_result() -> BaselineStability:
    # Coarse dt keeps this tractable; 3 yr, 1 spin-up.
    return assess_baseline(n_years=3, spinup_years=1, dt=7200.0)


@pytest.mark.slow
class TestPresentDayBaseline:

    def test_mean_pressure_is_roughly_six_millibar(self, baseline_result):
        assert 5.5 < baseline_result.mean_pressure_mb < 7.0  # ~6 mb present-day Mars

    def test_has_a_seasonal_swing(self, baseline_result):
        assert baseline_result.seasonal_swing_pa > 20.0  # CO2 cycle is present

    def test_is_a_stable_nondrifting_limit_cycle(self, baseline_result):
        assert abs(baseline_result.pressure_drift_pa_per_year) < 2.0
        assert baseline_result.swing_repeatability_pa < 5.0
        assert baseline_result.is_stable(drift_tol_pa_per_year=2.0)

    def test_temperature_and_ice_do_not_run_away(self, baseline_result):
        assert abs(baseline_result.temperature_drift_k_per_year) < 0.5
        assert 150.0 < baseline_result.mean_temperature_k < 280.0
        assert baseline_result.mean_ice_kg > 0.0

    def test_default_ice_reservoir_inflates_pressure(self):
        # The finding: the model's *default* 5e15 kg cap is not consistent with a
        # 6 mb atmosphere — it inflates the baseline well above 6 mb.
        inflated = assess_baseline(n_years=3, spinup_years=1, dt=7200.0, ice_mass=5.0e15)
        assert inflated.mean_pressure_mb > 6.8
