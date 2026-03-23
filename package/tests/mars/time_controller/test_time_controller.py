"""Unit tests for the TimeController itself.

These tests verify the TimeController's mechanics:
    • Snapshot creation
    • Elapsed time advancement
    • Callback invocation
    • Input validation (dt, duration)
    • Snapshot field types
    • Accuracy enum storage
    • Single-step evolve()
"""

from __future__ import annotations

import unittest

import torch

from src.celestials import Mars
from src.engine import Accuracy, TimeController


def _val(tensor: torch.Tensor) -> float:
    """Extract a Python float from a torch.Tensor."""
    return float(tensor.item())


class TestTimeController(unittest.TestCase):
    """Verify the TimeController drives evolution correctly."""

    def test_controller_creates_snapshots(self):
        """Running for a short duration should produce snapshots."""
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        history = tc.run(duration=3600.0 * 10)
        self.assertEqual(len(history), 10)

    def test_controller_elapsed_time_advances(self):
        """Elapsed time in snapshots should increase monotonically."""
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        history = tc.run(duration=3600.0 * 5)
        for i in range(1, len(history)):
            self.assertGreater(_val(history[i].time), _val(history[i - 1].time))

    def test_controller_callback(self):
        """Callback should be invoked at each step."""
        calls: list[float] = []
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        tc.run(duration=3600.0 * 3,
               callback=lambda state, t: calls.append(_val(t)))
        self.assertEqual(len(calls), 3)

    def test_invalid_dt_raises(self):
        """dt ≤ 0 should raise."""
        mars = Mars()
        with self.assertRaises(ValueError):
            TimeController(mars, dt=0)

    def test_invalid_duration_raises(self):
        """duration ≤ 0 should raise."""
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        with self.assertRaises(ValueError):
            tc.run(duration=-1)

    def test_snapshot_values_are_tensors(self):
        """Snapshot fields should be torch.Tensor."""
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        history = tc.run(duration=3600.0)
        snap = history[0]
        self.assertIsInstance(snap.time, torch.Tensor)
        self.assertIsInstance(snap.surface_temperature, torch.Tensor)
        self.assertIsInstance(snap.surface_pressure, torch.Tensor)

    def test_accuracy_enum_is_stored(self):
        """Controller should store the Accuracy enum."""
        mars = Mars()
        tc_fast = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        tc_acc = TimeController(mars, dt=3600.0, accuracy=Accuracy.ACCURATE)
        self.assertEqual(tc_fast.accuracy, Accuracy.FAST)
        self.assertEqual(tc_acc.accuracy, Accuracy.ACCURATE)

    def test_evolve_single_step(self):
        """A single evolve() call should advance the state."""
        mars = Mars()
        tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
        t0 = _val(mars.elapsed_time)
        tc.evolve(tc.dt)
        t1 = _val(mars.elapsed_time)
        self.assertGreater(t1, t0)


if __name__ == "__main__":
    unittest.main()
