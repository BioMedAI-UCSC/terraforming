"""Tests for src.gcm3d.coordinates (requires the optional 'gcm3d' extra).

Skipped when dinosaur/JAX are not installed (the default in the torch-only Tests
CI); the dedicated gcm3d CI workflow installs the extra and runs these. Covers:
  - grid / coordinate_system build valid dinosaur objects (planet-agnostic)
  - the pseudo-spectral transform underneath is differentiable (jax.grad flows)
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("dinosaur")

from src.gcm3d import coordinate_system, grid  # noqa: E402


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
