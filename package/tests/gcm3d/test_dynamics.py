"""Tests for src.gcm3d.dynamics — the Phase-1 go/no-go, as regression tests.

Requires the optional 'gcm3d' extra. Builds a dry model for Mars (via
MARS_BODY_3D) to exercise the planet-agnostic core with a real body. Covers:
  - a dry primitive-equations model builds and integrates STABLY (finite state
    over a multi-hour rollout)
  - the rollout is DIFFERENTIABLE: grad of a diagnostic w.r.t. the initial state
    is finite, nonzero, and matches finite differences

These are the two go/no-go conditions for the whole 3-D effort. Small
(T21 / 12 layers / tens of steps) to run in CI.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.gcm3d import (  # noqa: E402
    coordinate_system,
    integrate,
    physics_specs,
    primitive_equations,
    stepper,
)
from src.gcm3d._dinosaur import primitive_equations_states, scales  # noqa: E402

_u = scales.units


@pytest.fixture(scope="module")
def model():
    coords = coordinate_system("T21", n_layers=12)
    specs = physics_specs(MARS_BODY_3D)
    eq = primitive_equations(coords, MARS_BODY_3D, specs=specs)
    step = stepper(eq, dt_seconds=120.0, specs=specs)
    init_fn, _ = primitive_equations_states.steady_state_jw(
        coords, specs, p0=610 * _u.pascal, t0=200 * _u.kelvin, u0=20 * _u.meter / _u.second
    )
    return coords, step, init_fn()


class TestBuildAndIntegrate:

    def test_rollout_is_stable_and_finite(self, model):
        coords, step, state0 = model
        final = jax.jit(lambda s: integrate(step, s, 200))(state0)
        T = coords.horizontal.to_nodal(final.temperature_variation)
        lsp = coords.horizontal.to_nodal(final.log_surface_pressure)
        assert bool(jnp.all(jnp.isfinite(T)))
        assert bool(jnp.all(jnp.isfinite(lsp)))
        assert float(jnp.max(jnp.abs(T))) < 1e3  # bounded, no blow-up

    def test_integrate_rejects_nonpositive_steps(self, model):
        _, step, state0 = model
        with pytest.raises(ValueError):
            integrate(step, state0, 0)


class TestDifferentiable:
    """The reason to use this dycore: gradients flow through the dynamics."""

    def _loss_fn(self, model):
        coords, step, state0 = model

        def loss(alpha):
            s0 = dataclasses.replace(
                state0, temperature_variation=state0.temperature_variation * alpha
            )
            final = integrate(step, s0, 30)
            return jnp.sum(coords.horizontal.to_nodal(final.temperature_variation) ** 2)

        return loss

    def test_grad_through_rollout_is_finite_nonzero(self, model):
        g = jax.grad(self._loss_fn(model))(1.0)
        assert bool(jnp.isfinite(g))
        assert abs(float(g)) > 0.0

    def test_grad_matches_finite_difference(self, model):
        loss = self._loss_fn(model)
        g = jax.grad(loss)(1.0)
        eps = 1e-3
        fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
        assert float(g) == pytest.approx(float(fd), rel=1e-3)
