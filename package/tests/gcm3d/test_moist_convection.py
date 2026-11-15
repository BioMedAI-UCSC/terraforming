"""Moist adjustment limits, latent budgets and reverse derivatives."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.moist_convection import adjust
from src.framework.physics.saturation_pressure import specific_humidity

jax.config.update('jax_enable_x64',True)


def case():
    half=jnp.linspace(0.,50000.,13);full=.5*(half[1:]+half[:-1]);t=jnp.linspace(220.,300.,12)
    q=1.15*specific_humidity(t,full).specific_humidity
    return t,q,half,full


def test_dry_and_stable_supersaturated_columns_have_no_adjustment():
    half=jnp.linspace(0.,610.,5);full=.5*(half[1:]+half[:-1]);t=jnp.full(4,200.)
    for factor in (.5,1.15):
        q=factor*specific_humidity(t,full).specific_humidity
        result=jax.jit(lambda tt:adjust(tt,q,half,full,300.))(t)
        np.testing.assert_array_equal(result.temperature_tendency,jnp.zeros(4))
        np.testing.assert_array_equal(result.vapor_tendency,jnp.zeros(4))
        assert result.rain_kg_m2_s==0. and result.snow_kg_m2_s==0.
        assert result.diagnostics['iterations']==0


def test_capped_two_layer_adjustment_has_analytic_latent_heat_and_rain():
    half=jnp.array([0.,305.,610.]);full=.5*(half[1:]+half[:-1]);t=jnp.array([290.,330.])
    epsilon=189.02/461.5;q=jnp.full(2,epsilon*1.01);dt=300.
    result=jax.jit(lambda tt:adjust(tt,q,half,full,dt))(t)
    np.testing.assert_allclose(q+dt*result.vapor_tendency,epsilon,rtol=2e-15)
    np.testing.assert_allclose(jnp.mean(dt*result.temperature_tendency),
        2.5e6/735.94*epsilon*.01,rtol=2e-14)
    np.testing.assert_allclose(result.rain_kg_m2_s,epsilon*.01*610./3.72/dt,rtol=2e-14)
    assert result.snow_kg_m2_s==0. and not result.diagnostics['unconverged']
    limited=adjust(t,q,half,full,dt,iterations=1,tolerance_max=.02)
    assert limited.diagnostics['unconverged']
    np.testing.assert_allclose(limited.temperature_tendency,result.temperature_tendency,rtol=2e-14)


def test_moist_enthalpy_and_water_budget_and_batched_columns():
    t,q,half,full=case();dt=300.
    t=jnp.stack([t,t-30.],axis=-1);q=jnp.stack([q,
        1.15*specific_humidity(t[:,1],full).specific_humidity],axis=-1)
    half=jnp.broadcast_to(half[:,None],(13,2));full=jnp.broadcast_to(full[:,None],(12,2))
    result=jax.jit(lambda tt:adjust(tt,q,half,full,dt))(t)
    assert np.any(result.temperature_tendency!=0.)
    latent=jnp.where(t[-1]<273.15,2.834e6,2.5e6)
    mass=jnp.diff(half,axis=0)/3.72
    np.testing.assert_allclose(jnp.sum((735.94*result.temperature_tendency+
        latent*result.vapor_tendency)*mass,axis=0),0.,atol=1e-7)
    np.testing.assert_allclose(jnp.sum(result.vapor_tendency*mass,axis=0)+
        result.rain_kg_m2_s+result.snow_kg_m2_s,0.,atol=1e-14)
    assert result.rain_kg_m2_s[0]>0. and result.snow_kg_m2_s[1]>0.


def test_moist_reverse_derivative_matches_centered_difference():
    t,q,half,full=case()
    def objective(amplitude):
        result=adjust(t,q*amplitude,half,full,300.)
        return jnp.sum(result.temperature_tendency)+1e3*result.rain_kg_m2_s
    auto=jax.jit(jax.grad(objective))(1.);forward=jax.jit(objective);eps=1e-5
    fd=(forward(1.+eps)-forward(1.-eps))/(2.*eps)
    assert np.isfinite(auto) and abs(float(auto))>1e-8
    np.testing.assert_allclose(auto,fd,rtol=1e-7,atol=1e-9)


def test_invalid_moist_configuration_and_shapes_are_rejected():
    t,q,half,full=case()
    with pytest.raises(ValueError,match='positive time step'):
        adjust(t,q,half,full,0.)
    with pytest.raises(ValueError,match='ordered tolerances'):
        adjust(t,q,half,full,300.,tolerance_max=.001)
    with pytest.raises(ValueError,match='matching'):
        adjust(t,q[:-1],half,full,300.)
