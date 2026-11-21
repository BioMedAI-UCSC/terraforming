import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.rayleigh_drag import rayleigh_tendencies

jax.config.update('jax_enable_x64',True)


def test_surface_drag_timescale_units_sigma_limits_and_disabled_switch():
    p=jnp.array([.1,.7,.8,1.,1.1]);u=jnp.ones(5)*100.;v=-u
    days=rayleigh_tendencies(u,v,p,1.,0.,timescale=-1.)
    seconds=rayleigh_tendencies(u,v,p,1.,0.,timescale=86400.)
    for a,b in zip(days,seconds,strict=True): np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(days[0],[0.,0.,-100./(3.*86400.),-100./86400.,0.],atol=1e-17)
    for value in rayleigh_tendencies(u,v,p,1.,0.,timescale=0.):
        np.testing.assert_array_equal(value,0.)


def test_tropical_top_sponge_uses_mask_without_clipping_wind():
    p=jnp.full((3,2),1e-5);u=jnp.full_like(p,300.);v=u*.2
    mask=jnp.array([0.,.5])
    kwargs=dict(timescale=0.,model_top=True,tropical=True,mask=mask)
    first=rayleigh_tendencies(u,v,p,610.,jnp.array([0.,.2]),**kwargs)
    doubled=rayleigh_tendencies(2.*u,2.*v,p,610.,jnp.array([0.,.2]),**kwargs)
    for a,b in zip(first,doubled,strict=True):
        np.testing.assert_array_equal(a[:,0],0.)
        np.testing.assert_allclose(b,2.*a,rtol=2e-14)
        assert np.all(np.asarray(a[:,1])<0.)


def test_rayleigh_reverse_derivatives_reach_pressure_latitude_and_timescale():
    p=jnp.array([.01,.1,10.,500.])
    def objective(scale):
        du,dv=rayleigh_tendencies(jnp.full(4,50.),jnp.full(4,-20.),p*scale,610.,.2*scale,
            timescale=-scale,model_top=True,tropical=True)
        return jnp.sum(du+dv)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>0.
    for eps in (1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-10)
