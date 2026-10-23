"""Tests for framework GCM dynamics — the Phase-1 go/no-go regressions.

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

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.framework.gcm.coordinates import coordinate_system  # noqa: E402
from src.framework.gcm.dynamics import (  # noqa: E402
    horizontal_hyperdiffusion_filter, integrate, primitive_equations, stepper,
)
from src.framework.gcm.specs import physics_specs  # noqa: E402
from src.framework.gcm._dinosaur import primitive_equations_states, scales  # noqa: E402

_u = scales.units
jax.config.update("jax_enable_x64", True)


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


class TestHyperdiffusion:

    def test_filtered_stepper_matches_the_imex_step_followed_by_damping(self, model):
        coords, _, state = model
        specs = physics_specs(MARS_BODY_3D)
        equation = primitive_equations(coords, MARS_BODY_3D, specs=specs)
        state = dataclasses.replace(
            state,
            temperature_variation=state.temperature_variation.at[:, 0, -1].add(1e-3),
        )
        raw = jax.jit(stepper(equation, 120.0, specs))(state)
        filtered = jax.jit(stepper(
            equation, 120.0, specs, coords=coords, hyperdiffusion_tau_seconds=600.0,
        ))(state)
        np.testing.assert_allclose(
            filtered.temperature_variation[:, 0, -1],
            raw.temperature_variation[:, 0, -1] * np.exp(-0.2),
            rtol=1e-12, atol=1e-15,
        )
        np.testing.assert_array_equal(filtered.log_surface_pressure, raw.log_surface_pressure)

    def test_modal_decay_preserves_mean_pressure_tracers_and_surface_reservoirs(self, model):
        from src.framework.physics.gcm import ColumnPhysicsState

        coords, _, state = model
        grid = coords.horizontal
        spectrum = jnp.zeros_like(state.temperature_variation)
        spectrum = spectrum.at[:, 0, 0].set(3.0)
        spectrum = spectrum.at[:, 0, 1].set(2.0)
        spectrum = spectrum.at[:, 0, -1].set(1.0)
        state = dataclasses.replace(
            state, vorticity=spectrum, divergence=spectrum,
            temperature_variation=spectrum,
            tracers={"particle_mass": spectrum + 4.0}, sim_time=jnp.asarray(7.0),
        )
        surface = jnp.ones((1,) + grid.nodal_shape)
        # Include soil with a spectral-looking last dimension: surface state
        # must be excluded by its contract, not an accidental shape mismatch.
        column = ColumnPhysicsState(state, surface * 200.0, surface * 0.01, spectrum + 180.0)
        apply = horizontal_hyperdiffusion_filter(
            coords, dt_seconds=300.0, specs=physics_specs(MARS_BODY_3D), tau_seconds=600.0,
        )
        filtered = jax.jit(apply)(column, column)
        eigen = np.asarray(grid.laplacian_eigenvalues)
        expected = np.asarray(spectrum) * np.exp(-0.5 * (eigen / eigen[-1])**4)
        for name in ("vorticity", "divergence", "temperature_variation"):
            np.testing.assert_allclose(getattr(filtered.dynamics, name), expected, rtol=1e-12)
        assert float(filtered.dynamics.temperature_variation[0, 0, -1]) == pytest.approx(np.exp(-0.5))
        assert float(filtered.dynamics.temperature_variation[0, 0, 0]) == 3.0
        np.testing.assert_array_equal(filtered.dynamics.log_surface_pressure, state.log_surface_pressure)
        np.testing.assert_array_equal(filtered.dynamics.tracers["particle_mass"], state.tracers["particle_mass"])
        assert filtered.dynamics.sim_time == state.sim_time
        for name in ("surface_temperature", "co2_ice", "ground_temperature"):
            np.testing.assert_array_equal(getattr(filtered, name), getattr(column, name))

    def test_filter_gradient_matches_analytic_decay(self, model):
        coords, _, state = model
        apply = horizontal_hyperdiffusion_filter(coords, 300.0, physics_specs(MARS_BODY_3D), 600.0)

        def loss(amplitude):
            changed = dataclasses.replace(
                state, temperature_variation=jnp.zeros_like(state.temperature_variation).at[0, 0, -1].set(amplitude),
            )
            return apply(changed, changed).temperature_variation[0, 0, -1]

        assert float(jax.grad(loss)(1.0)) == pytest.approx(np.exp(-0.5))

    @pytest.mark.parametrize("tau", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_damping_time_is_rejected(self, model, tau):
        coords, _, _ = model
        with pytest.raises(ValueError, match="tau_seconds"):
            horizontal_hyperdiffusion_filter(coords, 300.0, physics_specs(MARS_BODY_3D), tau)

    def test_stepper_requires_coordinates_when_damping_is_requested(self):
        with pytest.raises(ValueError, match="coords"):
            stepper(None, 300.0, physics_specs(MARS_BODY_3D), hyperdiffusion_tau_seconds=600.0)
