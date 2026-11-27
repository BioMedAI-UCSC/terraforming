"""Legacy convection conservation away from its neutral branch boundary."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.legacy_convection import adjust

jax.config.update('jax_enable_x64',True)


def case(stable=False):
    half=jnp.array([0.,50.,150.,350.,610.]);full=.5*(half[1:]+half[:-1]);kap=189.02/735.94
    theta=jnp.array([350.,320.,280.,250.]) if stable else jnp.array([250.,280.,320.,350.])
    t=theta*(full/610.)**kap;u=jnp.arange(4.)*10.;v=-u
    r=jnp.arange(8.).reshape(4,2)*1e-5
    return t,u,v,r,half,full,kap


def test_legacy_unstable_column_preserves_heat_and_material_and_leaves_winds():
    values=case();t,u,v,r,half,full,kap=values
    result=jax.jit(adjust)(*values);mass=jnp.diff(half)/3.72
    np.testing.assert_allclose(jnp.sum(result[0]*mass),jnp.sum(t*mass),rtol=2e-15)
    np.testing.assert_allclose(jnp.sum(result[3]*mass[:,None],axis=0),jnp.sum(r*mass[:,None],axis=0),rtol=2e-15)
    for actual,expected in zip(result[1:3],(u,v),strict=True):
        np.testing.assert_array_equal(actual,expected)
    potential=result[0]/(full/610.)**kap
    assert np.max(np.abs(np.diff(potential)))<1e-12


def test_stable_legacy_column_and_reverse_derivative():
    t,u,v,r,half,full,kap=case(True)
    result=adjust(t,u,v,r,half,full,kap)
    np.testing.assert_allclose(result[0],t,rtol=2e-15)
    np.testing.assert_array_equal(result[3],r)
    t,u,v,r,half,full,kap=case()
    def objective(offset):
        changed=t.at[-1].add(offset)
        return jnp.sum(adjust(changed,u,v,r,half,full,kap)[0]**2)
    auto=jax.jit(jax.grad(objective))(0.);eps=1e-4
    fd=(objective(eps)-objective(-eps))/(2.*eps)
    assert auto>0.
    np.testing.assert_allclose(auto,fd,rtol=1e-8,atol=1e-5)


def test_legacy_shape_errors_are_rejected():
    t,u,v,r,half,full,kap=case()
    with pytest.raises(ValueError,match='matching'):
        adjust(t,u,v,r[:-1],half,full,kap)
