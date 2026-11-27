import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.framework.physics.co2_microphysics import (
    phase_change,particle_radius,cloud_step,LATENT_HEAT,HEAT_CAPACITY)

jax.config.update('jax_enable_x64',True)


def case():
    half=jnp.array([0.,10.,80.,250.,610.])
    full=.5*(half[:-1]+half[1:])
    frost=3182.48/(23.3494-jnp.log(full/100.))
    return half,full,frost


def test_phase_change_conserves_source_latent_energy_and_evaporates_available_ice():
    _,p,frost=case()
    t=frost+jnp.array([-5.,5.,30.,-2.])
    q=jnp.array([0.,.02,.001,-.001])
    new_t,new_q=phase_change(t,p,q)
    np.testing.assert_allclose(HEAT_CAPACITY*new_t-LATENT_HEAT*new_q,
        HEAT_CAPACITY*t-LATENT_HEAT*jnp.maximum(q,0.),atol=3e-11,rtol=2e-14)
    assert float(new_q[2])==0.
    assert np.min(new_q)>=0.
    np.testing.assert_allclose(new_t[new_q>0.],frost[new_q>0.],atol=1e-13)


@pytest.mark.parametrize('dt',[5.,300.,30000.])
@pytest.mark.parametrize('diffusivity',[0.,.001,100.])
def test_sedimentation_preserves_total_co2_with_signed_phase_exchange(dt,diffusivity):
    half,p,frost=case()
    q=jnp.array([.001,.02,.005,.003])
    result=jax.jit(cloud_step)(frost+jnp.array([-5.,5.,-2.,10.]),half,p,q,
        150.,jnp.full(3,diffusivity),dt)
    gas=jnp.diff(half)/3.72
    np.testing.assert_allclose(jnp.sum(gas*q),jnp.sum(gas*result.ice_ratio-
        result.gas_mass_loss_kg_m2)+result.deposition_kg_m2,rtol=3e-13,atol=1e-13)
    assert np.min(result.ice_ratio)>=0.
    assert float(result.deposition_kg_m2)>=0.


@pytest.mark.parametrize('dt',[5.,30000.])
@pytest.mark.parametrize('clear',[False,True])
def test_reverse_gradients_match_finite_differences_in_clear_and_cloudy_air(dt,clear):
    half,p,frost=case()
    t=frost+(20. if clear else -5.)
    q=jnp.zeros(4) if clear else jnp.array([.001,.005,.003,.02])
    def objective(scale):
        result=cloud_step(t+scale,half,p,q,150.,jnp.full(3,.001),dt)
        return jnp.sum(result.temperature)+100.*result.deposition_kg_m2+jnp.sum(result.ice_ratio)
    auto=jax.jit(jax.grad(objective))(0.)
    assert np.isfinite(auto)
    for eps in (1e-3,1e-4):
        finite=(objective(eps)-objective(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-7)
    assert np.isfinite(jax.grad(lambda mass:jnp.sum(particle_radius(mass)))(0.))


def test_batched_columns_match_independent_solves_with_distinct_switches():
    half,p,frost=case()
    temperatures=jnp.stack([frost-5.,frost+20.],axis=1)
    q=jnp.full((4,2),.005)
    result=jax.jit(cloud_step)(temperatures,jnp.broadcast_to(half[:,None],(5,2)),
        jnp.broadcast_to(p[:,None],(4,2)),q,jnp.array([150.,180.]),jnp.full((3,2),.001),300.)
    for i in range(2):
        single=cloud_step(temperatures[:,i],half,p,q[:,i],jnp.array([150.,180.])[i],jnp.full(3,.001),300.)
        for actual,expected in zip(result[:5],single[:5],strict=True):
            np.testing.assert_allclose(actual[...,i],expected,rtol=2e-13,atol=1e-13)


def test_reverse_gradients_reach_pressure_cloud_nuclei_and_mixing():
    half,p,frost=case()
    def objective(scale):
        result=cloud_step(frost-5.,half*scale,p*scale,jnp.full(4,.005),
            150.,jnp.full(3,100.)*scale,300.,nuclei_per_kg=1.e5*scale)
        return jnp.sum(result.temperature)+100.*result.deposition_kg_m2+jnp.sum(result.ice_ratio)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-5
    for eps in (1e-4,1e-5):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-7)


def test_cloud_grid_and_nuclei_validation():
    half,p,frost=case()
    with pytest.raises(ValueError,match='nuclei'):
        cloud_step(frost,half,p,jnp.zeros(4),150.,jnp.zeros(3),5.,nuclei_per_kg=0.)
    with pytest.raises(ValueError,match='diffusivity'):
        cloud_step(frost,half,p,jnp.zeros(4),150.,jnp.zeros(4),5.)


def test_cloud_density_uses_post_phase_temperature():
    half,p,frost=case()
    result=cloud_step(frost-5.,half,p,jnp.full(4,.001),150.,jnp.zeros(3),231.)
    np.testing.assert_allclose(result.diagnostics['density_kg_m3'],p/(189.02*result.temperature),rtol=2e-15)
    assert np.max(np.abs(result.temperature-(frost-5.)))>1.


def test_bottom_boundary_density_change_is_explicit_for_shared_water_solver():
    from src.framework.physics.co2_microphysics import sedimentation
    half,p,frost=case();t=frost-5.;post=frost
    args=(jnp.full(4,.001),half,p,t,150.,jnp.full(4,1e-6),jnp.zeros(3),231.)
    water=sedimentation(*args,density_temperature=post)
    co2=sedimentation(*args,density_temperature=post,bottom_density_temperature=post[-1])
    np.testing.assert_array_equal(water[2]['density_kg_m3'],co2[2]['density_kg_m3'])
    assert abs(float(water[1]-co2[1]))>1e-12
