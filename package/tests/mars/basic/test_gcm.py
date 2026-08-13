"""Tests for src/celestials/planets/mars/gcm.py :: MarsGCM construction.

Covers:
  - MarsGCM constructs through the Mars constructor (setup_properties hook).
  - It builds a JCM model and bootstraps the state without integrating.
  - It keeps the scalar diagnostic mirror from the base Mars class.
  - It honours Mars constructor kwargs.
  - step_gcm is still a stub (Step 4 wires it).

These tests build a JCM model, so they are slow.
"""

from __future__ import annotations

import pytest

from src.celestials.planets.mars.gcm import MarsGCM
from src.celestials.planets.mars.planet import Mars


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mars_gcm():
    """Build one MarsGCM for the module (construction is slow)."""
    return MarsGCM()


# ── Construction ──────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestMarsGCMConstruction:

    def test_is_mars_subclass(self, mars_gcm):
        """MarsGCM is a Mars, so it fits the Planet interface."""
        assert isinstance(mars_gcm, Mars)

    def test_builds_model(self, mars_gcm):
        """setup_properties builds a JCM model."""
        assert mars_gcm._model is not None

    def test_bootstraps_state_and_carry(self, mars_gcm):
        """bootstrap_state populates the dycore state and the physics carry."""
        assert mars_gcm._model._final_dycore_state is not None
        assert mars_gcm._model._final_physics_state is not None

    def test_builds_forcing(self, mars_gcm):
        """setup_properties stores a ForcingData object."""
        from jcm.forcing import ForcingData
        assert isinstance(mars_gcm._forcing, ForcingData)

    def test_keeps_scalar_mirror(self, mars_gcm):
        """The base Mars mirror survives; the default state is present."""
        assert float(mars_gcm.thermal.surface_temperature) == pytest.approx(210.0)
        assert float(mars_gcm.atmosphere.surface_pressure) == pytest.approx(610.0)

    def test_honours_constructor_kwargs(self):
        """A custom Mars kwarg reaches the scalar mirror."""
        planet = MarsGCM(surface_temperature=200.0)
        assert float(planet.thermal.surface_temperature) == pytest.approx(200.0)

    def test_step_gcm_updates_mirror(self):
        """step_gcm advances the model and writes finite scalars to the mirror."""
        import math
        import torch
        planet = MarsGCM()
        planet.step_gcm(torch.tensor(3600.0))   # one hour
        t = float(planet.thermal.surface_temperature)
        p = float(planet.atmosphere.surface_pressure)
        assert math.isfinite(t) and t > 0.0
        assert math.isfinite(p) and p > 0.0

    def test_uses_mola_terrain_when_present(self, mars_gcm):
        """When the MOLA file exists, the terrain has real relief."""
        import numpy as np
        from src.celestials.planets.mars.terrain import DEFAULT_MOLA_IMG
        if not DEFAULT_MOLA_IMG.exists():
            pytest.skip("MOLA data not downloaded")
        assert float(np.abs(np.asarray(mars_gcm._model.terrain.phis0)).max()) > 0.0


# ── Engine seam: drive MarsGCM through TimeController (Accuracy.GCM) ───────────

class TestTimeControllerSeam:

    def test_base_planet_step_gcm_raises(self):
        """A planet with no GCM backend raises on step_gcm."""
        import torch
        from src.celestials import Mars
        with pytest.raises(NotImplementedError):
            Mars().step_gcm(torch.tensor(3600.0))

    def test_accuracy_gcm_exists(self):
        """The engine exposes the GCM strategy."""
        from src.engine import Accuracy
        assert Accuracy.GCM.value == "gcm"

    @pytest.mark.slow
    def test_run_returns_snapshots(self):
        """TimeController with Accuracy.GCM drives MarsGCM and yields Snapshots."""
        import math
        from src.engine import TimeController, Accuracy
        tc = TimeController(MarsGCM(), dt=3600.0, accuracy=Accuracy.GCM)
        history = tc.run(duration=3600.0 * 3)   # three one-hour steps
        assert len(history) == 3
        for snap in history:
            assert math.isfinite(float(snap.surface_temperature))
            assert float(snap.surface_pressure) > 0.0
