import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.correlated_k import correlated_k_optical_depths,load_radiation_tables
from src.framework.physics.radiation_grid import radiation_grid
from src.framework.physics.radiative_transfer import column_fluxes
from src.framework.physics.particle_optics import co2_cloud_optics,make_particle_provider

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('mode',['15band','extended'])
@pytest.mark.parametrize('cloud',[False,True])
def test_extended_composition_energy_telescope_and_reverse_derivative(mode,cloud):
    half=jnp.array([0.,10.,80.,250.,610.]);full=.5*(half[:-1]+half[1:])
    t=jnp.array([145.,155.,170.,190.]);solar=jnp.asarray(load_radiation_tables(mode)['solar_weights'])*590.
    def fluxes(offset):
        grid=radiation_grid(half,full,t+offset,210.+offset)
        particles=co2_cloud_optics(jnp.full(4,.002),grid.pressure) if cloud else None
        return column_fluxes(t+offset,210.+offset,half,full,.6,solar,.2,.95,3.72,
            radiation_mode=mode,particle_optics=None if particles is None else particles[:2],
            water_mass_fraction=jnp.array([1e-7,1e-5,.005,.05]))
    result=jax.jit(fluxes)(0.)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result))
    np.testing.assert_allclose(jnp.sum(result.atmospheric_convergence_w_m2)+result.surface_net_w_m2,
        result.toa_net_down_w_m2,rtol=2e-13,atol=2e-12)
    objective=lambda offset:jnp.sum(fluxes(offset).longwave_up_w_m2)
    auto=jax.jit(jax.grad(objective))(.023)
    assert np.isfinite(auto) and abs(float(auto))>1e-5
    for epsilon in (1e-3,1e-4):
        finite=(objective(.023+epsilon)-objective(.023-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-7)


@pytest.mark.parametrize('mode,channels',[('12band',17),('15band',33),('extended',17)])
def test_all_water_mixture_bins_and_source_out_of_range_temperature_limits(mode,channels):
    t=jnp.array([10.,30.,950.,1000.]);p=jnp.full(4,610.)
    a=correlated_k_optical_depths(t,p,jnp.full(4,1.),mode=mode,water_mass_fraction=.005)
    b=correlated_k_optical_depths(t,p,jnp.full(4,1.),mode=mode,water_mass_fraction=.009)
    for first,second in zip(a,b,strict=True):
        assert first.shape[1]==channels
        np.testing.assert_array_equal(first[:,-1],0.)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[:,:,0],first[:,:,1])
        np.testing.assert_array_equal(first[:,:,2],first[:,:,3])


def test_particle_provider_rejects_incompatible_or_incomplete_configuration():
    with pytest.raises(ValueError,match='15-band'):
        make_particle_provider(co2_cloud_tracer='co2_cloud_ice')
    with pytest.raises(ValueError,match='mass and number'):
        make_particle_provider(dust_tracers=(('dust',),))
    with pytest.raises(ValueError,match='ice, core and number'):
        make_particle_provider(water_moment_tracers=('ice',))


def test_collision_absorption_is_connected_to_infrared_continuum():
    half=jnp.array([0.,1000.,10000.,30000.,50000.]);full=.5*(half[:-1]+half[1:])
    t=jnp.array([180.,210.,240.,270.])
    solar=jnp.asarray(load_radiation_tables('extended')['solar_weights'])*590.
    def fluxes(hydrogen):
        return column_fluxes(t,290.,half,full,.6,solar,.2,.95,3.72,
            radiation_mode='extended',cia_molar_fractions=(.9,hydrogen))
    baseline=column_fluxes(t,290.,half,full,.6,solar,.2,.95,3.72,radiation_mode='extended')
    actual=jax.jit(fluxes)(.1)
    np.testing.assert_allclose(actual.shortwave_down_w_m2,baseline.shortwave_down_w_m2,rtol=2e-14)
    assert np.max(np.abs(actual.longwave_down_w_m2-baseline.longwave_down_w_m2))>1e-3
    np.testing.assert_allclose(jnp.sum(actual.atmospheric_convergence_w_m2)+actual.surface_net_w_m2,
        actual.toa_net_down_w_m2,rtol=2e-13,atol=2e-12)
    objective=lambda h:jnp.sum(fluxes(h).longwave_down_w_m2)
    auto=jax.jit(jax.grad(objective))(.1)
    for eps in (1e-4,1e-5):
        finite=(objective(.1+eps)-objective(.1-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-6)
