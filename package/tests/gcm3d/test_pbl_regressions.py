"""Independent stability, neutral-column and enthalpy checks for PBL mixing."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import reference_temperature
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics
from src.framework.gcm._dinosaur import primitive_equations, scales

jax.config.update("jax_enable_x64", True)


def setup_column():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(BODY)
    grid = coords.horizontal
    zeros = jnp.zeros((4,) + grid.modal_shape)
    ps = float(specs.nondimensionalize(610 * scales.units.pascal))
    dyn = primitive_equations.State(
        vorticity=zeros, divergence=zeros, temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps))),
        sim_time=0.,
    )
    return coords, specs, physics.initial_column_state(dyn, coords, 160., specs)


def test_pbl_exchange_uses_actual_cold_air(monkeypatch):
    coords, specs, state = setup_column()
    captured = []
    original = physics._surface_exchange_properties

    def exchange(*args, **kwargs):
        captured.append(np.asarray(args[5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(physics, "_surface_exchange_properties", exchange)
    air_variation = jnp.full((4,) + coords.horizontal.nodal_shape, -50.)
    physics.pbl_vertical_diffusion_tendencies(
        state, coords, specs, BODY, dataclasses.replace(radiative_forcing(), pbl_diffusion_enabled=True), temperature_nodal=air_variation)
    np.testing.assert_allclose(captured[0], 150.)


@pytest.mark.parametrize("neutral", [True, False])
def test_pbl_neutral_stationarity_and_enthalpy_with_jit_grad(neutral):
    coords, specs, state = setup_column()
    grid = coords.horizontal
    sigma = jnp.asarray(coords.vertical.centers, dtype=jnp.float64)
    exner = sigma ** BODY.kappa
    ref = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    theta = jnp.full((4,), 220.) if neutral else jnp.array([260., 245., 225., 215.])
    weights = jnp.asarray(np.diff(coords.vertical.boundaries))[:, None, None]
    # Zero winds isolate heat diffusion from dissipative momentum heating.
    winds = (jnp.zeros((4,) + grid.nodal_shape),) * 2

    def tendency(profile):
        air = jnp.broadcast_to((profile * exner)[:, None, None], (4,) + grid.nodal_shape)
        return grid.to_nodal(physics.pbl_vertical_diffusion_tendencies(
            state, coords, specs, BODY, dataclasses.replace(radiative_forcing(), pbl_diffusion_enabled=True),
            temperature_nodal=air-ref, wind_nodal=winds)[2])

    heat = jax.jit(tendency)(theta)
    assert np.isfinite(heat).all()
    # Momentum softplus smoothing contributes < 1e-10 K/s here.
    assert np.max(np.abs(np.sum(np.asarray(weights * heat), axis=0))) < 1e-10
    if neutral:
        assert np.max(np.abs(heat)) < 1e-10
    else:
        assert np.max(np.abs(heat)) > 1e-9
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(tendency(p)**2)))(theta)
    assert np.isfinite(gradient).all()
