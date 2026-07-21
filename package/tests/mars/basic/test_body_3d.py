"""Tests for MARS_BODY_3D in src.celestials.planets.mars (pure Python).

Runs in the standard torch-only Tests CI. Verifies the Mars body spec for the
3-D GCM core is consistent with the existing MARS_* constants (single source of
truth) and carries the right CO2 thermodynamics.
"""

from __future__ import annotations

import math

import pytest

from src.celestials.planets.mars import (
    MARS_BODY_3D,
    MARS_GRAVITY,
    MARS_RADIUS,
    MARS_ROTATION_PERIOD,
)
from src.gcm3d.body import BodyConstants


class TestMarsBody3D:

    def test_is_body_constants(self):
        assert isinstance(MARS_BODY_3D, BodyConstants)
        assert MARS_BODY_3D.name == "Mars"

    def test_reuses_existing_mars_constants(self):
        # Single source of truth: derived from the MARS_* module constants.
        assert MARS_BODY_3D.radius_m == pytest.approx(float(MARS_RADIUS), rel=1e-12)
        assert MARS_BODY_3D.gravity_m_s2 == pytest.approx(float(MARS_GRAVITY), rel=1e-12)
        assert MARS_BODY_3D.rotation_period_s == pytest.approx(
            float(MARS_ROTATION_PERIOD), rel=1e-12
        )

    def test_angular_velocity_is_one_sol(self):
        assert MARS_BODY_3D.angular_velocity_s == pytest.approx(
            2 * math.pi / float(MARS_ROTATION_PERIOD), rel=1e-12
        )

    def test_kappa_is_co2(self):
        # R/cp for CO2 (~0.245), below Earth-air's 2/7.
        assert MARS_BODY_3D.kappa == pytest.approx(188.92 / 770.0, rel=1e-9)
        assert MARS_BODY_3D.kappa < 2.0 / 7.0

    def test_reference_pressure_is_present_day_mars(self):
        assert MARS_BODY_3D.reference_surface_pressure_pa == pytest.approx(610.0)
