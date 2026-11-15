"""Restart decoding preserves a geographic landmark, SI wind and physical time."""
import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyse_reviewer_fields import checkpoint_fields
from audit_paper_results import paired_timestep
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.framework.gcm._dinosaur import jax, jnp, primitive_equations, scales, spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.restart import save_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import initial_column_state


def test_restart_landmark_wind_units_and_weighted_local_error(tmp_path):
    jax.config.update("jax_enable_x64", True)
    coords, specs = coordinate_system("T21", 4), physics_specs(BODY)
    grid = coords.horizontal
    shape = (4,) + grid.nodal_shape
    pattern = (1.2 * np.sin(grid.latitudes)[None]
               + .7 * np.cos(grid.longitudes)[:, None] * np.cos(grid.latitudes)[None])
    temperature = jnp.broadcast_to(pattern[None], shape)
    velocity = float(specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
    wind = jnp.broadcast_to(10. * jnp.cos(grid.latitudes)[None, None], shape)
    vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(grid, wind / velocity, jnp.zeros(shape))
    pressure = float(specs.nondimensionalize(610. * scales.units.pascal))
    dynamics = primitive_equations.State(vorticity=vor, divergence=div,
        temperature_variation=grid.to_modal(temperature),
        log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(pressure))), sim_time=0.)
    initial = initial_column_state(dynamics, coords, BODY.reference_temperature_k, specs)
    ndsecond = float(specs.nondimensionalize(1. * scales.units.second))
    final = initial._replace(dynamics=dataclasses.replace(initial.dynamics, sim_time=300. * ndsecond,
        temperature_variation=grid.to_modal(temperature + 2.)))
    paths = [save_restart(initial, tmp_path / "initial.npz"), save_restart(final, tmp_path / "final.npz")]
    data = checkpoint_fields(paths, [0., 300.], coords, specs, BODY, {})
    np.testing.assert_allclose(data.air_temperature[0, 0], BODY.reference_temperature_k + pattern.T, atol=2e-5)
    np.testing.assert_allclose(data.eastward_wind[0, 0], np.asarray(wind[0]).T, atol=2e-5)
    assert data.eastward_wind.attrs["units"] == "m/s"
    reference = data.copy(deep=True)
    reference["air_temperature"][1] = reference.air_temperature[0]
    _, report = paired_timestep(data, reference, first_dt=300., second_dt=150.)
    assert report["fields"]["air_temperature"]["time_integrated_rmse"] == pytest.approx(np.sqrt(2.), abs=2e-5)
    with pytest.raises(ValueError, match="physical time"):
        checkpoint_fields(paths, [0., 600.], coords, specs, BODY, {})
