"""Tests for mars_gcm3d_core in src.celestials.planets.mars (needs 'gcm3d' extra).

Verifies the "core is callable from mars" path: mars_gcm3d_core wires
MARS_BODY_3D into the planet-agnostic gcm3d core and returns a runnable dry Mars
model. Skipped when dinosaur/JAX are absent (torch-only CI).
"""

from __future__ import annotations

import pytest

pytest.importorskip("dinosaur")
import jax.numpy as jnp  # noqa: E402

from src.celestials.planets.mars import mars_gcm3d_core  # noqa: E402
from src.gcm3d import integrate, stepper  # noqa: E402


def test_mars_gcm3d_core_builds_runnable_model():
    coords, specs, equations = mars_gcm3d_core("T21", n_layers=10)
    # Mars nondimensional constants reached the specs.
    assert float(specs.radius) == pytest.approx(1.0, rel=1e-9)
    assert coords.nodal_shape == (10, 64, 32)

    # And the returned equations actually integrate one step (finite state).
    from src.gcm3d._dinosaur import primitive_equations_states, scales

    _u = scales.units
    init_fn, _ = primitive_equations_states.steady_state_jw(
        coords, specs, p0=610 * _u.pascal, t0=200 * _u.kelvin
    )
    step = stepper(equations, dt_seconds=120.0, specs=specs)
    final = integrate(step, init_fn(), 3)
    assert bool(jnp.all(jnp.isfinite(coords.horizontal.to_nodal(final.temperature_variation))))
