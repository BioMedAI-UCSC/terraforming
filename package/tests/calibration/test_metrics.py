"""Tests for src.calibration.metrics.

Covers the seasonal-cycle comparison metrics on synthetic, analytically-known
inputs (fast, deterministic — no model run).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.metrics import (
    amplitude_ratio,
    annual_mean,
    bias,
    compute_metrics,
    peak_to_peak_amplitude,
    phase_lag_deg,
    rmse,
)

LS = np.arange(0.0, 360.0, 30.0)


class TestScalarMetrics:

    def test_annual_mean(self):
        assert annual_mean(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)

    def test_peak_to_peak_amplitude(self):
        assert peak_to_peak_amplitude(np.array([1.0, 5.0, 2.0])) == pytest.approx(4.0)

    def test_rmse_zero_for_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        # errors [1, -1] -> sqrt(mean([1,1])) = 1
        assert rmse(np.array([2.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)

    def test_bias_is_signed(self):
        assert bias(np.array([3.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(2.0)

    def test_amplitude_ratio(self):
        model = np.array([0.0, 2.0])      # amp 2
        reference = np.array([0.0, 4.0])  # amp 4
        assert amplitude_ratio(model, reference) == pytest.approx(0.5)

    def test_amplitude_ratio_raises_on_flat_reference(self):
        with pytest.raises(ValueError):
            amplitude_ratio(np.array([0.0, 1.0]), np.array([2.0, 2.0]))


class TestPhaseLag:

    def test_zero_lag_for_aligned_cycles(self):
        ref = np.cos(np.radians(LS))
        assert phase_lag_deg(LS, ref.copy(), ref) == pytest.approx(0.0, abs=1e-9)

    def test_detects_known_shift(self):
        ref = np.cos(np.radians(LS))
        model = np.roll(ref, 2)  # shifted later by 2 bins = 60 deg
        assert phase_lag_deg(LS, model, ref) == pytest.approx(60.0, abs=1e-9)

    def test_wraps_to_signed_range(self):
        ref = np.cos(np.radians(LS))
        model = np.roll(ref, 11)  # +330 deg -> should wrap to -30
        assert phase_lag_deg(LS, model, ref) == pytest.approx(-30.0, abs=1e-9)


class TestComputeMetrics:

    def test_bundles_all_fields(self):
        ref = 780.0 + 100.0 * np.cos(np.radians(LS))
        model = ref + 20.0  # pure +20 bias, same shape
        m = compute_metrics(LS, model, ref)
        assert m.bias == pytest.approx(20.0)
        assert m.amplitude_ratio == pytest.approx(1.0)
        assert m.phase_lag_deg == pytest.approx(0.0, abs=1e-9)
        assert m.rmse == pytest.approx(20.0)
        assert m.reference_mean == pytest.approx(780.0)
