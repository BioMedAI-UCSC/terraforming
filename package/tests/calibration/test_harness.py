"""Tests for src.calibration.harness.

Two layers:
  - Fast, deterministic tests of the plumbing (override validation, field
    selection, and the calibration loop via an injected stub simulator).
  - Slow tests (`@pytest.mark.slow`) that actually run the 0-D model over a Mars
    year and validate against VL1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.harness import (
    PARAMETERS,
    SeasonalCycle,
    calibrate,
    evaluate,
    simulate_seasonal_cycle,
)
from src.calibration.reference import get_reference


# ── Fast: plumbing with a stub simulator ──────────────────────────────────────

class TestCalibrationLoop:
    """The Nelder-Mead loop, exercised without the model via an injected sim."""

    def _stub_factory(self, reference, optimum=0.05, scale=4000.0):
        # Model pressure = reference + linear miss that vanishes at `optimum`,
        # so the calibrator should drive the tuned parameter toward `optimum`.
        def stub(ref, *, overrides=None, **kw):
            cap = (overrides or {}).get("polar_cap_fraction", 0.0)
            offset = (cap - optimum) * scale
            return SeasonalCycle(
                ls_deg=ref.ls_deg.copy(),
                temperature_k=np.zeros_like(ref.ls_deg),
                pressure_pa=ref.pressure_pa + offset,
            )
        return stub

    def test_calibrate_reduces_rmse_and_finds_optimum(self):
        ref = get_reference("vl1")
        res = calibrate(
            ref, ["polar_cap_fraction"], x0=[0.01], field="pressure",
            simulate=self._stub_factory(ref, optimum=0.05), maxiter=100,
        )
        assert res.n_parameters == 1
        assert res.tuned_parameters["polar_cap_fraction"] == pytest.approx(0.05, abs=1e-3)
        assert res.rmse_after < res.rmse_before
        assert res.rmse_after == pytest.approx(0.0, abs=1e-1)

    def test_calibrate_reports_before_and_after_metrics(self):
        ref = get_reference("vl1")
        res = calibrate(
            ref, ["polar_cap_fraction"], x0=[0.01],
            simulate=self._stub_factory(ref), maxiter=50,
        )
        assert res.metrics_before.rmse == pytest.approx(res.rmse_before)
        assert res.metrics_after.rmse == pytest.approx(res.rmse_after)

    def test_calibrate_rejects_unknown_parameter(self):
        ref = get_reference("vl1")
        with pytest.raises(KeyError):
            calibrate(ref, ["not_a_param"], x0=[1.0], simulate=self._stub_factory(ref))

    def test_calibrate_rejects_x0_length_mismatch(self):
        ref = get_reference("vl1")
        with pytest.raises(ValueError):
            calibrate(ref, ["polar_cap_fraction"], x0=[0.01, 0.02],
                      simulate=self._stub_factory(ref))


class TestFieldSelection:

    def test_evaluate_rejects_bad_field(self):
        ref = get_reference("vl1")
        cyc = SeasonalCycle(ref.ls_deg.copy(), np.zeros(12), ref.pressure_pa.copy())
        with pytest.raises(ValueError):
            evaluate(cyc, ref, field="humidity")

    def test_evaluate_rejects_missing_reference_field(self):
        ref = get_reference("vl1")  # temperature_k is None
        cyc = SeasonalCycle(ref.ls_deg.copy(), np.zeros(12), ref.pressure_pa.copy())
        with pytest.raises(ValueError):
            evaluate(cyc, ref, field="temperature")

    def test_parameters_map_to_cached_attrs(self):
        # Guard the public knobs the calibration exposes.
        assert set(PARAMETERS) >= {"polar_cap_fraction", "thermal_inertia"}


# ── Slow: real model run against VL1 ──────────────────────────────────────────

@pytest.mark.slow
class TestAgainstVikingLander1:

    def test_seasonal_cycle_shape_and_physical(self):
        ref = get_reference("vl1")
        cyc = simulate_seasonal_cycle(ref, dt=3600.0, n_years=2)
        assert cyc.pressure_pa.shape == cyc.ls_deg.shape == ref.ls_deg.shape
        assert np.all(np.isfinite(cyc.pressure_pa))
        assert np.all(cyc.temperature_k > 100.0)  # physical Mars surface T
        assert np.all(cyc.pressure_pa > 0.0)

    def test_calibrating_cap_fraction_improves_amplitude(self):
        # The headline finding: the default cap fraction under-predicts the CO2
        # pressure swing; raising it recovers the amplitude and cuts RMSE.
        ref = get_reference("vl1")
        default = evaluate(simulate_seasonal_cycle(ref, dt=3600.0, n_years=2), ref)
        tuned = evaluate(
            simulate_seasonal_cycle(ref, overrides={"polar_cap_fraction": 0.05},
                                    dt=3600.0, n_years=2),
            ref,
        )
        assert default.amplitude_ratio < 0.3        # default badly under-predicts
        assert tuned.amplitude_ratio > default.amplitude_ratio
        assert tuned.rmse < default.rmse
