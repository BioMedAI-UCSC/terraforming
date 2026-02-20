"""Unit tests for Mars simulation – daily (one-sol) variations.

These tests verify that the simulation produces physically plausible
variations in temperature and pressure over one Martian sol (~88 775 s).

The ``Accuracy`` enum on the ``TimeController`` selects the integration
strategy:
    ``ACCURATE`` → RK4 (engine-owned integrator)
    ``FAST``     → reduced-order model (planet-owned analytic physics)

Expected physical behaviour (from NASA Mars Fact Sheet):
    • Diurnal temperature swing: tens of K (up to ~100 K)
    • Temperature range       : ~130 K (winter pole) to ~300 K (summer equator)
"""

from __future__ import annotations

import unittest

import tensorflow as tf

from src.celestials import Mars, MARS_ROTATION_PERIOD
from src.engine import Accuracy, TimeController, Snapshot


def _val(tensor: tf.Tensor) -> float:
    """Extract a Python float from a tf.Tensor."""
    return float(tensor.numpy())


class TestMarsDaily(unittest.TestCase):
    """Variations across one Martian sol."""

    def _run_one_sol(
        self, accuracy: Accuracy, dt: float = 900.0,
    ) -> list[Snapshot]:
        """Helper: run Mars for 1 sol, return snapshot history."""
        mars = Mars()
        tc = TimeController(mars, dt=dt, accuracy=accuracy)
        return tc.run(duration=MARS_ROTATION_PERIOD)

    # --- Accuracy.ACCURATE ---
    def test_accurate_temperature_varies_over_sol(self):
        """Temperature should change across the sol (not stay constant)."""
        history = self._run_one_sol(Accuracy.ACCURATE)
        temps = [_val(s.surface_temperature) for s in history]
        self.assertGreater(max(temps) - min(temps), 0.0,
                           "Temperature should vary across one sol")

    def test_accurate_pressure_nonnegative(self):
        """Pressure must remain ≥ 0 at all times."""
        history = self._run_one_sol(Accuracy.ACCURATE)
        for s in history:
            self.assertGreaterEqual(_val(s.surface_pressure), 0.0)

    def test_accurate_temperature_physical_range(self):
        """Temperature should stay in a plausible Martian range (50-400 K)."""
        history = self._run_one_sol(Accuracy.ACCURATE)
        for s in history:
            t = _val(s.surface_temperature)
            self.assertGreater(t, 50.0, f"T={t} K too low")
            self.assertLess(t, 400.0, f"T={t} K too high")

    # --- Accuracy.FAST ---
    def test_fast_temperature_varies_over_sol(self):
        """Fast model: temperature should also show variation."""
        history = self._run_one_sol(Accuracy.FAST)
        temps = [_val(s.surface_temperature) for s in history]
        self.assertGreater(max(temps) - min(temps), 0.0,
                           "Temperature should vary across one sol (fast)")

    def test_fast_pressure_nonnegative(self):
        """Pressure must remain ≥ 0 at all times (fast model)."""
        history = self._run_one_sol(Accuracy.FAST)
        for s in history:
            self.assertGreaterEqual(_val(s.surface_pressure), 0.0)

    def test_fast_temperature_physical_range(self):
        """Temperature should stay in a plausible Martian range (50-400 K)."""
        history = self._run_one_sol(Accuracy.FAST)
        for s in history:
            t = _val(s.surface_temperature)
            self.assertGreater(t, 50.0)
            self.assertLess(t, 400.0)


if __name__ == "__main__":
    unittest.main()
