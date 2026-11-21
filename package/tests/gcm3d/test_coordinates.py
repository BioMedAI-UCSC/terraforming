"""Tests for src.framework.gcm.coordinates (requires the optional 'gcm3d' extra).

Skipped when dinosaur/JAX are not installed (the default in the torch-only Tests
CI); the dedicated gcm3d CI workflow installs the extra and runs these. Covers:
  - grid / coordinate_system build valid dinosaur objects (planet-agnostic)
  - the pseudo-spectral transform underneath is differentiable (jax.grad flows)
"""

from __future__ import annotations

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("dinosaur")

from src.framework.gcm.coordinates import coordinate_system, grid  # noqa: E402


class TestCoordinateSystem:

    def test_grid_builds_for_known_truncations(self):
        for t in ("T21", "T42", "T85"):
            assert grid(t) is not None

    def test_grid_rejects_unknown_truncation(self):
        with pytest.raises(ValueError):
            grid("T999")

    def test_coordinate_system_shape(self):
        coords = coordinate_system("T21", n_layers=20)
        # nodal shape = (layers, lon_nodes, lat_nodes); T21 -> 64 x 32
        assert coords.nodal_shape == (20, 64, 32)
        assert coords.modal_shape[0] == 20

    def test_t42_shape(self):
        coords = coordinate_system("T42", n_layers=25)
        assert coords.nodal_shape == (25, 128, 64)

    def test_rejects_nonpositive_layers(self):
        with pytest.raises(ValueError):
            coordinate_system("T21", n_layers=0)

    def test_explicit_sigma_levels_are_retained(self):
        boundaries=np.array([0.,.001,.1,.7,1.])
        coords=coordinate_system('T21',4,sigma_boundaries=boundaries)
        np.testing.assert_array_equal(coords.vertical.boundaries,boundaries)
        np.testing.assert_array_equal(coords.vertical.centers,.5*(boundaries[:-1]+boundaries[1:]))

    @pytest.mark.parametrize('boundaries',[[0.,.5,1.],[.01,.1,.5,.7,1.],
        [0.,.1,.1,.7,1.],[0.,.1,np.nan,.7,1.]])
    def test_invalid_explicit_sigma_levels_are_rejected(self,boundaries):
        with pytest.raises(ValueError,match='sigma boundaries'):
            coordinate_system('T21',4,sigma_boundaries=boundaries)


class TestDifferentiableTransform:
    """The reason to use this dycore: gradients flow through the spectral core."""

    def test_grad_flows_through_nodal_transform(self):
        g = grid("T21")
        modal = jax.random.uniform(jax.random.PRNGKey(0), g.modal_shape)

        def loss(m):
            return jnp.sum(g.to_nodal(m) ** 2)

        grad = jax.grad(loss)(modal)
        assert grad.shape == modal.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
