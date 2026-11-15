import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.framework.physics.boundary_layer import layer_heights,surface_exchange,eddy_coefficients,column_step

jax.config.update("jax_enable_x64",True)


def test_isothermal_geometry_and_temperature_response():
    half=jnp.array([0.,50.,150.,350.,610.])[:,None]
    full=(half[1:]+half[:-1])*.5
    temperature=jnp.full((4,1),200.)
    mid,interface=layer_heights(temperature,half,full,189.02,3.72)
    np.testing.assert_allclose(mid,189.02*200./3.72*np.log(610./full),rtol=2e-15)
    np.testing.assert_allclose(interface,189.02*200./3.72*np.log(610./half[1:-1]),rtol=2e-15)
    cold,_=layer_heights(temperature*.5,half,full,189.02,3.72)
    np.testing.assert_allclose(cold,mid*.5,rtol=2e-15)
    derivative=jax.grad(lambda scale:jnp.sum(layer_heights(temperature*scale,
        half,full,189.02,3.72)[0]))(1.)
    np.testing.assert_allclose(derivative,np.sum(mid),rtol=2e-15)


@pytest.mark.parametrize("theta_ground",[180.,220.])
def test_surface_exchange_reverse_derivative(theta_ground):
    def objective(wind):
        cdh,cdm,ustar,thstar,_,_=surface_exchange(200.,theta_ground,wind,3.,50.,.01,3.72)
        return cdh+cdm+ustar+thstar
    auto=jax.grad(objective)(10.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(10.+eps)-objective(10.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-9)


def test_zero_wind_keeps_reference_forward_value_and_finite_subgradient():
    def stress(u):
        _,cdm,ustar,_,_,_=surface_exchange(200.,210.,u,0.,50.,.01,3.72)
        return cdm*u*u+ustar
    assert float(stress(0.))==0.
    assert np.isfinite(jax.grad(stress)(0.))


def test_eddy_coefficients_retain_shear_memory_and_have_reverse_derivative():
    upper=jnp.array([[10.,3.,210.],[20.,2.,210.]])
    lower=jnp.array([[20.,5.,200.],[30.,5.,220.]])
    def evaluate(scale,previous):
        return eddy_coefficients(upper,lower*scale,jnp.array([.1,.2]),jnp.array([.2,.3]),
            jnp.array([-.01,-.02]),jnp.array([100.,50.]),previous,300.,3.72)
    previous=jnp.array([.01,.02])
    km,kh,ri,memory=evaluate(1.,previous)
    instantaneous=jnp.sum(((lower-upper)*jnp.array([-.01,-.02])[:,None])[:,:2]**2,axis=-1)
    np.testing.assert_allclose(memory,previous-(previous-instantaneous)*.03,rtol=2e-15)
    assert np.all(np.asarray(km)>=.001) and np.all(np.asarray(kh)>=.001)
    assert np.max(np.abs(evaluate(1.,jnp.zeros(2))[0]-km))>0.
    def objective(scale):
        outputs=evaluate(scale,previous)
        return sum(jnp.sum(x) for x in outputs)
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-9)


def test_eddy_minimum_is_the_ames_floor_at_zero_shear():
    values=jnp.array([[0.,0.,200.]])
    def coefficient(scale):
        return eddy_coefficients(values,values*scale,jnp.ones(1),jnp.ones(1),
            -jnp.ones(1),jnp.ones(1)*100.,jnp.zeros(1),300.,3.72)[0].sum()
    np.testing.assert_allclose(coefficient(1.),.001,rtol=0.,atol=0.)
    assert np.isfinite(jax.grad(coefficient)(1.))


@pytest.mark.parametrize("surface_k",[180.,230.])
def test_full_column_conserves_closed_gas_tracers_and_is_differentiable(surface_k):
    half=jnp.array([0.,50.,150.,350.,610.])[:,None]
    full=(half[1:]+half[:-1])*.5
    temperature=jnp.array([160.,180.,200.,210.])[:,None]
    wind=jnp.array([30.,20.,10.,5.])[:,None]
    tracer=jnp.arange(4,dtype=jnp.float64).reshape(4,1,1)*1e-5
    def advance(scale):
        return column_step(temperature,wind*scale,-wind,tracer,half,full,surface_k,
            300.,189.02,735.94,3.72)
    result=jax.jit(advance)(1.)
    dp=np.diff(half,axis=0)[...,None]
    np.testing.assert_allclose(np.sum(tracer*dp,axis=0),np.sum(result[3]*dp,axis=0),rtol=2e-13)
    assert np.min(np.asarray(result[3]))>=0.
    def objective(scale):
        t,u,v,r,d=advance(scale)
        return jnp.sum(t*t)+jnp.sum(u*u)+jnp.sum(v*v)+jnp.sum(r*r)+jnp.sum(d['heat_flux_w_m2'])
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-6)


def test_full_column_zero_wind_has_finite_stress_derivative():
    half=jnp.array([0.,50.,150.,350.,610.])
    full=(half[1:]+half[:-1])*.5
    temperature=jnp.array([160.,180.,200.,210.])
    def objective(wind):
        result=column_step(temperature,jnp.full(4,wind),jnp.zeros(4),jnp.empty((4,0)),
            half,full,220.,300.,189.02,735.94,3.72)
        return jnp.sum(result[0])+result[4]['stress_x_pa']+result[4]['stress_y_pa']
    assert np.isfinite(objective(0.))
    assert np.isfinite(jax.grad(objective)(0.))
