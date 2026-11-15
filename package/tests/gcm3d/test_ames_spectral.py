import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.ames_spectral import visible_spectrum,infrared_spectrum,_channel_selection

jax.config.update('jax_enable_x64',True)


def inputs(bands):
    dtau=jnp.full((3,bands,17),.2)
    cumulative=jnp.broadcast_to(jnp.array([0.,0.,.1,.2,.3,.4,.6])[:,None,None],(7,bands,17))
    return dtau,jnp.full_like(dtau,.3),jnp.full_like(dtau,.4),cumulative


def test_reference_threshold_equalities_and_nearly_clear_band_switch():
    active,weights=_channel_selection(jnp.array([[.001,.0009],[1.,1.]]),
        jnp.array([.8,.99]),jnp.array([.4,.6]),.001)
    np.testing.assert_array_equal(active,[[True,False],[False,False]])
    np.testing.assert_allclose(weights,[[.08,0.,.92],[0.,0.,.99]],atol=1e-15)


@pytest.mark.parametrize('kind',['visible','infrared'])
def test_spectral_reverse_derivatives_through_opacity_and_temperature(kind):
    dtau,w,g,cumulative=inputs(7 if kind=='visible' else 5)
    gas=jnp.full((dtau.shape[1],16),.1)
    def objective(scale):
        if kind=='visible':
            result=visible_spectrum(dtau*scale,w,g,cumulative*scale,gas*scale,
                .6,jnp.arange(1.,8.)*20.,.2)
        else:
            result=infrared_spectrum(dtau*scale,w,g,cumulative*scale,gas*scale,
                jnp.geomspace(.001,6.1,7),jnp.linspace(150.03,240.03,7)*scale,.1)
        return sum(jnp.sum(x) for x in result)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for eps in (1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-6)


@pytest.mark.parametrize('kind',['visible','infrared'])
def test_weak_gas_channels_are_replaced_by_clear_channel(kind):
    dtau,w,g,cumulative=inputs(7 if kind=='visible' else 5)
    gas=jnp.zeros((dtau.shape[1],16))
    def run(opacity):
        if kind=='visible':
            return visible_spectrum(opacity,w,g,cumulative,gas,.6,jnp.ones(7)*50.,.2)
        return infrared_spectrum(opacity,w,g,cumulative,gas,
            jnp.geomspace(.001,6.1,7),jnp.linspace(150.,240.,7),.1)
    expected=run(dtau)
    # Excluded channels can contain invalid unused optics without poisoning AD.
    altered=run(dtau.at[:,:,:-1].set(jnp.nan))
    for a,b in zip(altered,expected,strict=True):
        np.testing.assert_allclose(a,b,rtol=2e-14)


@pytest.mark.parametrize('kind',['visible','infrared'])
def test_spectral_map_columns_match_independent_columns(kind):
    dtau,w,g,cumulative=inputs(7 if kind=='visible' else 5)
    factors=jnp.array([.8,1.2])
    arrays=[x[...,None]*factors for x in (dtau,w,g,cumulative)]
    gas=jnp.full((dtau.shape[1],16,2),.1)
    pressure=jnp.geomspace(.001,6.1,7)[:,None]*factors
    temperature=jnp.linspace(150.,240.,7)[:,None]*factors
    def run(values,gas,p,t,mu,reflectivity):
        if kind=='visible':
            return visible_spectrum(*values,gas,mu,jnp.ones(7)*50.,reflectivity)
        return infrared_spectrum(*values,gas,p,t,reflectivity,scale_surface_flux=True)
    mu=jnp.array([.4,.7]);reflectivity=jnp.array([.1,.2])
    batch=jax.jit(run)(arrays,gas,pressure,temperature,mu,reflectivity)
    for i in range(2):
        single=run([x[...,i] for x in arrays],gas[...,i],pressure[:,i],temperature[:,i],mu[i],reflectivity[i])
        for a,b in zip(batch,single,strict=True):
            np.testing.assert_allclose(a[...,i],b,rtol=2e-12,atol=1e-10)
