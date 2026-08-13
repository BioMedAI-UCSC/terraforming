"""Tests for src/celestials/planets/mars/gcm.py :: _apply_mars_constants.

Covers:
  - The bridge applies the canonical MARS.as_jcm_overrides() map into JCM.
  - Mars planetary constants (radius, gravity, rotation, cp, kappa) are set.
  - Universal constants (Stefan-Boltzmann, von Kármán) stay at Earth values.
  - Water constants (alhc, tmelt) stay at Earth values (dry CO2 atmosphere).
  - Physical invariants: Mars gravity and radius are less than Earth's.

Note: set_constants writes a JCM global singleton.  No other package test uses
JCM, so the mutation does not affect them.
"""

from __future__ import annotations

import pytest

from src.celestials.planets.mars.constants import MARS
from src.celestials.planets.mars.gcm import _apply_mars_constants


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def applied():
    """Apply the Mars constants once and return the active PhysicalConstants."""
    return _apply_mars_constants()


# ── _apply_mars_constants ─────────────────────────────────────────────────────

class TestApplyMarsConstants:

    def test_applies_every_canonical_override(self, applied):
        """Each field from MARS.as_jcm_overrides() lands in JCM."""
        for field, expected in MARS.as_jcm_overrides().items():
            actual = getattr(applied, field)
            assert actual == pytest.approx(expected, rel=1e-9), field

    def test_sets_mars_planetary_values(self, applied):
        """The planetary constants match the Mars fact sheet."""
        assert applied.rearth == pytest.approx(MARS.radius_m)
        assert applied.grav == pytest.approx(MARS.gravity_m_s2)
        assert applied.omega == pytest.approx(MARS.rotation_rate_rad_s)
        assert applied.cpd == pytest.approx(MARS.co2_cp_j_kg_k)
        assert applied.akap == pytest.approx(MARS.kappa)

    def test_leaves_universal_constants_at_earth_values(self, applied):
        """Universal constants are not overridden."""
        assert applied.sbc == pytest.approx(5.67e-08)
        assert applied.karman_const == pytest.approx(0.4)

    def test_leaves_water_constants_at_earth_values(self, applied):
        """Water constants stay at Earth values; Mars is a dry CO2 world."""
        # alhc (water condensation) and tmelt (water melt) are not in the map.
        assert "alhc" not in MARS.as_jcm_overrides()
        assert "tmelt" not in MARS.as_jcm_overrides()
        assert applied.alhc == pytest.approx(2501000.0)
        assert applied.tmelt == pytest.approx(273.15)

    def test_returns_physical_constants_instance(self, applied):
        """The bridge returns the active PhysicalConstants object."""
        from jcm import constants as jcm_const
        assert isinstance(applied, jcm_const.PhysicalConstants)

    # ── Invariants ────────────────────────────────────────────────────────────

    def test_mars_is_smaller_and_weaker_than_earth(self, applied):
        """Mars gravity and radius are less than Earth's defaults."""
        assert applied.grav < 9.81
        assert applied.rearth < 6371000.0

    def test_solar_flux_is_less_than_one_au(self, applied):
        """Mars is farther from the Sun, so its flux is below the 1 AU value."""
        assert applied.solc < 1361.0
