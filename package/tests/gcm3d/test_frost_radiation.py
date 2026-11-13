"""Ames snow optics and Kirchhoff's surface infrared boundary."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm._dinosaur import scales
from src.framework.physics import gcm as physics

jax.config.update("jax_enable_x64", True)


def test_ames_snow_threshold_and_hemisphere_properties():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(BODY)
    shape = coords.horizontal.nodal_shape
    scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    mass = jnp.zeros(shape).at[0].set(2.).at[1].set(2.01)
    state = SimpleNamespace(co2_ice=(mass * BODY.gravity_m_s2 / scale)[None])
    forcing = radiative_forcing()
    result = physics.surface_radiative_forcing(state, coords, specs, BODY, forcing)
    north = np.asarray(coords.horizontal.latitudes) > 0.
    np.testing.assert_allclose(result.albedo[0], forcing.albedo)
    np.testing.assert_allclose(result.emissivity[0], forcing.emissivity)
    np.testing.assert_allclose(result.albedo[1], np.where(north, .65, .43))
    np.testing.assert_allclose(result.emissivity[1], np.where(north, .81, .88))
    disabled = dataclasses.replace(forcing, co2_frost_radiative_properties=False)
    assert physics.surface_radiative_forcing(state, coords, specs, BODY, disabled) is disabled


@pytest.mark.parametrize("value", [-.1, 1.1, np.nan])
def test_invalid_snow_optics_are_rejected(value):
    forcing = dataclasses.replace(radiative_forcing(), co2_frost_albedo_north=value)
    with pytest.raises(ValueError, match="co2_frost_albedo_north"):
        physics.surface_radiative_forcing(None, None, None, BODY, forcing)


@pytest.mark.parametrize("ames", [False, True])
@pytest.mark.parametrize("emissivity", [0., .81, 1.])
def test_surface_absorbs_only_emissivity_times_downward_infrared(ames, emissivity):
    forcing = dataclasses.replace(radiative_forcing(),
        ames_correlated_k_enabled=ames, emissivity=emissivity)
    surface_k = jnp.full((1, 1), 150.)
    flux = physics.column_radiative_fluxes(
        jnp.full((4, 1, 1), 180.), surface_k, jnp.full((1, 1), 610.),
        jnp.zeros((1, 1)), jnp.ones((1, 1)), .3, .1,
        np.linspace(0., 1., 5), BODY, forcing)
    expected = emissivity * (forcing.stefan_boltzmann * surface_k**4
                            - flux.longwave_down_w_m2[-1])
    np.testing.assert_allclose(flux.longwave_up_w_m2[-1] - flux.longwave_down_w_m2[-1],
                               expected, rtol=2.e-13, atol=1.e-13)
    np.testing.assert_allclose(jnp.sum(flux.atmospheric_convergence_w_m2, axis=0)
                               + flux.surface_net_w_m2, flux.toa_net_down_w_m2, atol=1.e-12)
