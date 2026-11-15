"""Moment-water material, empty-cell and reverse-gradient checks."""
import numpy as np
import pytest

pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.water_bulk import saturation_ratio
from src.framework.physics.water_moments import nucleation_condensation,shape_factor,subcycled_cloud_step

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('modes',[1,2])
@pytest.mark.parametrize('latent',[False,True])
@pytest.mark.parametrize('saturation',[.01,2.,10.])
def test_water_and_dust_transfer_close(modes,latent,saturation):
    t=190.;p=610.;dt=100.
    q=jnp.array([1e-5,1e8,1e-6,1e6,1e-8,saturation_ratio(t,p)*saturation])
    dm=jnp.array([1e-5,2e-5])[:modes];dn=jnp.array([1e8,1.5e8])[:modes]
    step=jax.jit(lambda q:nucleation_condensation(t,p,q,dt,dust_mass=dm,dust_number=dn,latent_heat=latent))
    result=step(q);dq=result.tracer_tendency
    np.testing.assert_allclose(dq[2]+dq[5],0.,atol=1e-22)
    if modes==1:
        np.testing.assert_allclose(dq[0]+dq[4],0.,atol=1e-22)
        np.testing.assert_allclose(dq[1]+dq[3],0.,atol=1e-9)
    else:
        np.testing.assert_allclose(jnp.sum(result.dust_mass_tendency)+dq[4],0.,atol=1e-22)
        np.testing.assert_allclose(jnp.sum(result.dust_number_tendency)+dq[3],0.,atol=1e-9)
    assert np.all(q+dt*dq>=0.)
    if latent:
        np.testing.assert_allclose(result.temperature_tendency*735.94,dq[2]*2.8343e6,atol=1e-16)


@pytest.mark.parametrize('latent',[False,True])
def test_reverse_gradient_matches_finite_differences(latent):
    q=jnp.array([1e-5,1e8,1e-6,1e6,1e-8,saturation_ratio(190.,610.)*3.])
    def loss(x):
        result=nucleation_condensation(x[0],x[1],q.at[5].multiply(x[2]),100.,latent_heat=latent)
        return result.tracer_tendency[2]*1e8+result.temperature_tendency*1e2
    x=jnp.array([190.,610.,1.]);grad=jax.jit(jax.grad(loss))(x)
    fd=[]
    for i,h in enumerate([1e-4,1e-3,1e-5]):
        fd.append((loss(x.at[i].add(h))-loss(x.at[i].add(-h)))/(2.*h))
    assert np.all(np.isfinite(grad)) and np.all(np.abs(grad)>0.)
    np.testing.assert_allclose(grad,fd,rtol=2e-5,atol=1e-10)


def test_empty_cells_and_batched_multimode_differentiation():
    def loss(x):
        q=jnp.zeros((2,6)).at[1].set(jnp.array([1e-5,1e8,1e-6,1e6,1e-8,1e-5])*x)
        result=nucleation_condensation(jnp.full(2,190.),jnp.full(2,610.),q,100.,
            dust_mass=jnp.stack([q[:,0],q[:,0]],axis=-1),
            dust_number=jnp.stack([q[:,1],q[:,1]],axis=-1),latent_heat=True)
        return jnp.sum(result.tracer_tendency[:,2])
    assert np.isfinite(jax.jit(jax.grad(loss))(1.))
    zero=nucleation_condensation(190.,610.,jnp.zeros(6),100.,latent_heat=True)
    for value in zero[:4]: np.testing.assert_array_equal(value,jnp.zeros_like(value))


def test_shape_factor_planar_limit_and_gradient():
    m=.965
    np.testing.assert_allclose(shape_factor(m,3000.01),(2.+m)*(1.-m)**2/4.,rtol=1e-14)
    assert np.isfinite(jax.grad(lambda x:shape_factor(m,x))(1000.))


def test_invalid_moment_shapes_are_rejected():
    with pytest.raises(ValueError,match='six entries'):
        nucleation_condensation(190.,610.,jnp.zeros(5),100.)
    with pytest.raises(ValueError,match='supplied together'):
        nucleation_condensation(190.,610.,jnp.zeros(6),100.,dust_mass=jnp.ones(1))


def test_subcycle_ramps_other_physics_and_has_reverse_gradient():
    q0=jnp.array([1e-5,1e8,1e-6,1e6,1e-8,saturation_ratio(190.,610.)*3.])
    q=q0.at[0].multiply(.9).at[5].multiply(1.2)
    def loss(t):
        result=subcycled_cloud_step(190.,t,610.,q0,q,100.,microtimestep=20.,latent_heat=True)
        return result.tracer_tendency[2]*1e8+result.temperature_tendency*1e2
    grad=jax.jit(jax.grad(loss))(191.)
    fd=(loss(191.0001)-loss(190.9999))/.0002
    np.testing.assert_allclose(grad,fd,rtol=2e-5,atol=1e-9)
    result=subcycled_cloud_step(190.,191.,610.,q0,q,100.,microtimestep=20.,latent_heat=True)
    np.testing.assert_allclose(result.tracer_tendency[2]+result.tracer_tendency[5],0.,atol=1e-22)
    one=subcycled_cloud_step(190.,191.,610.,q0,q,100.,microtimestep=-1.)
    direct=nucleation_condensation(191.,610.,q,100.)
    for actual,expected in zip(one,direct):
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-11)
