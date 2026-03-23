"""Unit tests for Mars simulation – yearly (one Martian year) variations.

These tests verify that the simulation produces physically plausible
variations over one Martian year (~5.94 × 10⁷ s, ~668.6 sols).

Expected physical behaviour (from NASA Mars Fact Sheet):
    • Seasonal pressure swing : ~100-200 Pa (CO₂ condensation cycle)
    • Temperature range       : ~130 K (winter pole) to ~300 K (summer equator)
    • Global mean T           : ~210 K
"""

from __future__ import annotations

import unittest

import torch

from src.celestials import Mars, MARS_ORBITAL_PERIOD
from src.engine import Accuracy, TimeController, Snapshot


def _val(tensor: torch.Tensor) -> float:
    """Extract a Python float from a torch.Tensor."""
    return float(tensor.item())


class TestMarsYearly(unittest.TestCase):
    """Variations across one Martian year."""

    def _run_one_year(
        self, accuracy: Accuracy, dt: float = 3600.0,
    ) -> list[Snapshot]:
        """Helper: run Mars for 1 Martian year."""
        mars = Mars()
        tc = TimeController(mars, dt=dt, accuracy=accuracy)
        return tc.run(duration=MARS_ORBITAL_PERIOD)

    # --- Accuracy.FAST (yearly) ---
    def test_fast_yearly_temperature_variation(self):
        """Temperature should show seasonal variation over a year."""
        history = self._run_one_year(Accuracy.FAST)
        temps = [_val(s.surface_temperature) for s in history]
        swing = max(temps) - min(temps)
        self.assertGreater(swing, 1.0,
                           f"Yearly temperature swing {swing:.2f} K too small")

    def test_fast_yearly_pressure_stays_reasonable(self):
        """Pressure should stay in a physically plausible range over a year."""
        history = self._run_one_year(Accuracy.FAST)
        for s in history:
            p = _val(s.surface_pressure)
            self.assertGreaterEqual(p, 0.0)
            self.assertLess(p, 1e6, "Pressure runaway detected")

    def test_fast_yearly_solar_flux_variation(self):
        """Solar flux should vary due to orbital eccentricity."""
        history = self._run_one_year(Accuracy.FAST)
        fluxes = [_val(s.solar_flux) for s in history]
        flux_swing = max(fluxes) - min(fluxes)
        self.assertGreater(flux_swing, 10.0,
                           "Solar flux should show eccentricity variation")

    def test_fast_orbital_angle_wraps(self):
        """Orbital angle should wrap around 2π over one full year."""
        history = self._run_one_year(Accuracy.FAST)
        angles = [_val(s.orbital_angle) for s in history]
        self.assertGreater(max(angles), 5.0,
                           "Orbital angle should approach 2π")

    # --- Accuracy.ACCURATE (yearly) ---
    def test_accurate_yearly_temperature_variation(self):
        """Accurate model: seasonal temperature swing over a year."""
        history = self._run_one_year(Accuracy.ACCURATE, dt=7200.0)
        temps = [_val(s.surface_temperature) for s in history]
        swing = max(temps) - min(temps)
        self.assertGreater(swing, 1.0,
                           f"Yearly temperature swing {swing:.2f} K too small")

    def test_accurate_yearly_pressure_stays_physical(self):
        """Pressure should remain positive and < ~1 MPa over a year."""
        history = self._run_one_year(Accuracy.ACCURATE, dt=7200.0)
        for s in history:
            p = _val(s.surface_pressure)
            self.assertGreaterEqual(p, 0.0)
            self.assertLess(p, 1e6, "Pressure runaway detected")


if __name__ == "__main__":
    unittest.main()
