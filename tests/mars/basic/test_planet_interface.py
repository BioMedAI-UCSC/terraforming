"""Basic unit tests for Mars types and framework interface.

These tests verify:
    • Mars is a proper Planet subclass
    • PlanetaryState behaves correctly (copy, values, tensor types)
    • Physical constants are torch.Tensor with float64 dtype
    • pack_state / unpack_state roundtrips
    • compute_derivatives returns correct shape and dtype
"""

from __future__ import annotations

import unittest

import torch

from src.constants import _c
from src.celestials import Mars


def _val(tensor: torch.Tensor) -> float:
    """Extract a Python float from a torch.Tensor."""
    return float(tensor.item())


class TestPlanetInterface(unittest.TestCase):
    """Verify the abstract interface contract."""

    def test_mars_is_planet(self):
        """Mars should be an instance of Planet."""
        from src.celestials import Planet
        mars = Mars()
        self.assertIsInstance(mars, Planet)

    def test_state_copy(self):
        """PlanetaryState.copy() should produce an independent clone."""
        mars = Mars()
        original_t = _val(mars.state.surface_temperature)
        clone = mars.state.copy()
        clone.surface_temperature = clone.surface_temperature + 100.0
        self.assertAlmostEqual(_val(mars.state.surface_temperature), original_t)

    def test_initial_state_values(self):
        """Default Mars state should have known values."""
        mars = Mars()
        self.assertAlmostEqual(_val(mars.state.surface_temperature), 210.0)
        self.assertAlmostEqual(_val(mars.state.surface_pressure), 610.0)
        self.assertIn("CO2", mars.state.composition)

    def test_custom_initial_conditions(self):
        """Mars should accept custom initial conditions."""
        mars = Mars(surface_temperature=250.0, surface_pressure=700.0)
        self.assertAlmostEqual(_val(mars.state.surface_temperature), 250.0)
        self.assertAlmostEqual(_val(mars.state.surface_pressure), 700.0)

    def test_state_values_are_tensors(self):
        """All state scalars should be torch.Tensor instances."""
        mars = Mars()
        self.assertIsInstance(mars.state.surface_temperature, torch.Tensor)
        self.assertIsInstance(mars.state.surface_pressure, torch.Tensor)
        self.assertIsInstance(mars.state.ice_mass, torch.Tensor)
        self.assertIsInstance(mars.state.albedo, torch.Tensor)
        self.assertIsInstance(mars.state.elapsed_time, torch.Tensor)

    def test_constants_are_tensors(self):
        """Physical constants should be torch.Tensor."""
        from src.constants import (
            STEFAN_BOLTZMANN, BOLTZMANN_K, G_NEWTON, AU_METRES,
        )
        for const in [STEFAN_BOLTZMANN, BOLTZMANN_K, G_NEWTON, AU_METRES]:
            self.assertIsInstance(const, torch.Tensor)
            self.assertEqual(const.dtype, torch.float64)

    def test_pack_unpack_state(self):
        """pack_state / unpack_state should roundtrip correctly."""
        mars = Mars()
        y = mars.pack_state()
        self.assertEqual(y.shape, (3,))
        self.assertAlmostEqual(float(y[0].item()), 210.0)
        # Modify and unpack
        y_new = y + 1.0
        mars.unpack_state(y_new)
        self.assertAlmostEqual(_val(mars.state.surface_temperature), 211.0)

    def test_compute_derivatives_returns_tensor(self):
        """compute_derivatives should return a shape-[3] tensor."""
        mars = Mars()
        # Need solar flux to be set for meaningful derivatives
        mars.advance_orbit(_c(3600.0))
        y = mars.pack_state()
        dy = mars.compute_derivatives(y)
        self.assertEqual(dy.shape, (3,))
        self.assertEqual(dy.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
