"""Tests for src.gcm3d.specs (requires the optional 'gcm3d' extra).

Covers the planet-agnostic "constants into the equations" seam:
  - a body's physics_specs nondimensionalises to the grid convention
    (radius=1, angular_velocity=0.5) so PrimitiveEquationsSigma accepts it
  - kappa preserved; gravity/gas-constant positive
  - works for two different bodies (Mars, Earth) — genericity
"""

from __future__ import annotations

import pytest

pytest.importorskip("dinosaur")

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.gcm3d.body import EARTH  # noqa: E402
from src.gcm3d.specs import nondimensionalization_scale, physics_specs  # noqa: E402


@pytest.mark.parametrize("body", [MARS_BODY_3D, EARTH], ids=["mars", "earth"])
class TestPhysicsSpecs:

    def test_radius_nondimensionalises_to_one(self, body):
        # The grid's radius defaults to 1; the equations reject a mismatch.
        assert float(physics_specs(body).radius) == pytest.approx(1.0, rel=1e-9)

    def test_angular_velocity_half_convention(self, body):
        # time scale = 1/(2Ω) -> Ω nondim = 0.5
        assert float(physics_specs(body).angular_velocity) == pytest.approx(0.5, rel=1e-9)

    def test_kappa_preserved(self, body):
        assert float(physics_specs(body).kappa) == pytest.approx(body.kappa, rel=1e-6)

    def test_gravity_and_gas_constant_positive(self, body):
        specs = physics_specs(body)
        assert float(specs.gravity_acceleration) > 0.0
        assert float(specs.ideal_gas_constant) > 0.0

    def test_scale_is_body_anchored(self, body):
        import dinosaur

        one_radius = body.radius_m * dinosaur.scales.units.meter
        scale = nondimensionalization_scale(body)
        assert float(scale.nondimensionalize(one_radius)) == pytest.approx(1.0, rel=1e-9)
