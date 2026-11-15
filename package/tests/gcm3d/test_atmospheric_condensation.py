"""Layer condensation must heat air and transfer the same mass to frost."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import initial_rest_state
from src.framework.gcm._dinosaur import scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import integrate, reference_temperature, stepper
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics

jax.config.update("jax_enable_x64", True)
_u = scales.units


@pytest.fixture(scope="module")
def column():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(BODY)
    pressure = jnp.full(coords.horizontal.nodal_shape, 610.)
    cf = co2_forcing()
    return coords, specs, pressure, cf


def cold_air(coords, pressure, deficit=10.):
    frost = physics.co2_frost_point_k(jnp.asarray(coords.vertical.centers, dtype=pressure.dtype)[:, None, None] * pressure)
    reference = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    return frost - deficit - reference


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_condensation_timescale_is_rejected(column, value):
    coords, specs, _, cf = column
    with pytest.raises(ValueError, match="atmospheric_condensation_timescale_s"):
        physics.forced_co2_primitive_equations(coords, BODY, radiative_forcing(),
            dataclasses.replace(cf, atmospheric_condensation_timescale_s=value), specs=specs)
