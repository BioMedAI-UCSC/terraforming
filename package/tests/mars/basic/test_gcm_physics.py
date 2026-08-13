"""Tests for src/celestials/planets/mars/gcm_physics.py.

Covers:
  - mars_physics() returns a ComposablePhysics with one MarsHeldSuarez term.
  - MarsHeldSuarez is a HeldSuarez subclass with a Mars identity.
  - The Mars defaults are colder than the Earth Held-Suarez defaults.
  - The term drives a JCM model one step without NaNs (slow, integration).

Note: MarsHeldSuarez reads the JCM constants singleton at construction, so each
test applies the Mars constants first.
"""

from __future__ import annotations

import pytest

from src.celestials.planets.mars.gcm import _apply_mars_constants
from src.celestials.planets.mars.gcm_physics import MarsHeldSuarez, mars_physics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mars_constants():
    """Apply the Mars constants so the terms non-dimensionalize correctly."""
    return _apply_mars_constants()


# ── mars_physics factory ──────────────────────────────────────────────────────

class TestMarsPhysicsFactory:

    def test_returns_composable_physics(self, mars_constants):
        """The factory returns a ComposablePhysics object."""
        from jcm.physics.composable_physics import ComposablePhysics
        assert isinstance(mars_physics(), ComposablePhysics)

    def test_has_single_mars_term(self, mars_constants):
        """The physics holds exactly one Mars placeholder term."""
        phys = mars_physics()
        assert len(phys.terms) == 1
        assert isinstance(phys.terms[0], MarsHeldSuarez)
        assert phys.terms[0].name == "mars_held_suarez"


# ── MarsHeldSuarez term ───────────────────────────────────────────────────────

class TestMarsHeldSuarez:

    def test_is_held_suarez_subclass(self, mars_constants):
        """The Mars term reuses the JCM Held-Suarez form."""
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarez
        assert issubclass(MarsHeldSuarez, HeldSuarez)

    def test_colder_than_earth_defaults(self, mars_constants):
        """Mars peak and floor temperatures are below Earth's defaults."""
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarez
        mars = MarsHeldSuarez()
        earth = HeldSuarez()
        # Both non-dimensionalize against the same (Mars) constants, so the
        # order of the stored values matches the order of the SI inputs.
        assert float(mars.maxT.get_value()) < float(earth.maxT.get_value())
        assert float(mars.minT.get_value()) < float(earth.minT.get_value())


# ── Integration: drive a model one step ───────────────────────────────────────

class TestDrivesModel:

    @pytest.mark.slow
    def test_one_step_has_no_nan(self, mars_constants):
        """The Mars physics steps a JCM model and stays finite."""
        import jax.numpy as jnp
        from jcm.model import Model
        from jcm.terrain import TerrainData
        from jcm.physics.held_suarez.utils import get_held_suarez_coords

        coords = get_held_suarez_coords()
        terrain = TerrainData.from_coords(coords)
        model = Model(coords=coords, terrain=terrain, time_step=180,
                      physics=mars_physics())
        preds = model.run(save_interval=0.25, total_time=0.25)
        assert not bool(jnp.any(jnp.isnan(preds.dynamics.temperature)))
