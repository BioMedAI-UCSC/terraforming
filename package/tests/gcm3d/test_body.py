"""Tests for src.gcm3d.body (pure Python — no dinosaur/JAX required).

Runs in the standard torch-only Tests CI. Covers the planet-agnostic
``BodyConstants`` abstraction:
  - derived properties (angular velocity, kappa, surface area)
  - validation rejects non-physical values
  - the Earth reference body is internally consistent
  - genericity: two different bodies produce different constants
"""

from __future__ import annotations

import math

import pytest

from src.gcm3d.body import EARTH, BodyConstants


def _body(**overrides):
    base = dict(
        name="Test",
        radius_m=3.0e6,
        gravity_m_s2=3.7,
        rotation_period_s=88_000.0,
        gas_constant_j_kg_k=189.0,
        cp_j_kg_k=770.0,
    )
    base.update(overrides)
    return BodyConstants(**base)


class TestDerivedProperties:

    def test_angular_velocity_from_rotation_period(self):
        b = _body(rotation_period_s=88_775.244)
        assert b.angular_velocity_s == pytest.approx(2 * math.pi / 88_775.244, rel=1e-12)

    def test_kappa_is_r_over_cp(self):
        b = _body(gas_constant_j_kg_k=189.0, cp_j_kg_k=770.0)
        assert b.kappa == pytest.approx(189.0 / 770.0, rel=1e-12)

    def test_surface_area(self):
        b = _body(radius_m=3.3895e6)
        assert b.surface_area_m2 == pytest.approx(4 * math.pi * 3.3895e6**2, rel=1e-12)

    def test_frozen(self):
        with pytest.raises(Exception):
            _body().radius_m = 1.0  # type: ignore[misc]


class TestValidation:

    @pytest.mark.parametrize(
        "field", ["radius_m", "gravity_m_s2", "rotation_period_s", "gas_constant_j_kg_k", "cp_j_kg_k"]
    )
    def test_rejects_nonpositive(self, field):
        with pytest.raises(ValueError):
            _body(**{field: 0.0})
        with pytest.raises(ValueError):
            _body(**{field: -1.0})


class TestEarthReference:

    def test_earth_kappa_near_two_sevenths(self):
        # Dry air: R/cp ~ 2/7 ~ 0.286.
        assert EARTH.kappa == pytest.approx(2.0 / 7.0, rel=0.02)

    def test_earth_rotation_is_sidereal_day(self):
        assert EARTH.angular_velocity_s == pytest.approx(7.292e-5, rel=1e-3)


class TestGenericity:
    """The core is planet-agnostic: distinct bodies yield distinct constants."""

    def test_two_bodies_differ(self):
        mars_like = _body(name="Mars", radius_m=3.3895e6, gravity_m_s2=3.72076)
        assert mars_like.radius_m != EARTH.radius_m
        assert mars_like.gravity_m_s2 != EARTH.gravity_m_s2
        # CO2 kappa (~0.245) is below Earth-air's (~0.286)
        assert mars_like.kappa < EARTH.kappa
