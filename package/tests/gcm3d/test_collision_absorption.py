import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.collision_absorption import collision_induced_depths,interpolate_coefficients,load_tables

jax.config.update('jax_enable_x64',True)


def inputs():
    return jnp.array([0.,0.,0.,100.,200.,300.,400.]),jnp.array([80.,90.,100.,140.,180.,220.,260.])


def test_collision_absorption_obeys_density_squared_and_molar_scaling():
    p,t=inputs()
    base=collision_induced_depths(p,t)
    doubled=collision_induced_depths(p*2.,t)
    diluted=collision_induced_depths(p,t,co2_molar_fraction=.5)
    np.testing.assert_allclose(doubled,4.*base,rtol=3e-14)
    np.testing.assert_allclose(diluted,.25*base,rtol=3e-14)
    np.testing.assert_array_equal(base[:2],0.)
    assert np.min(base)>=0. and np.max(base)>0.


def test_hydrogen_contribution_is_linear_and_zero_bands_remain_finite():
    p,t=inputs();carbon=.9
    a=collision_induced_depths(p,t,co2_molar_fraction=carbon,h2_molar_fraction=0.)
    b=collision_induced_depths(p,t,co2_molar_fraction=carbon,h2_molar_fraction=.05)
    c=collision_induced_depths(p,t,co2_molar_fraction=carbon,h2_molar_fraction=.1)
    np.testing.assert_allclose(c-a,2.*(b-a),rtol=3e-14,atol=1e-16)
    coefficient=interpolate_coefficients(load_tables()['co2_translational'],jnp.array([20.,273.,1000.]))
    assert np.isfinite(coefficient).all()
    assert np.any(np.asarray(coefficient)==0.)


def test_source_temperature_clamp_applies_to_density_and_path_length():
    p,t=inputs()
    cold=collision_induced_depths(p,jnp.full_like(t,20.))
    floor=collision_induced_depths(p,jnp.full_like(t,100.))
    hot=collision_induced_depths(p,jnp.full_like(t,1000.))
    ceiling=collision_induced_depths(p,jnp.full_like(t,800.))
    np.testing.assert_array_equal(cold,floor)
    np.testing.assert_array_equal(hot,ceiling)


def test_collision_reverse_derivatives_reach_temperature_pressure_and_hydrogen():
    p,t=inputs()
    def objective(scale):
        return jnp.sum(collision_induced_depths(p*scale,t+scale,
            co2_molar_fraction=.9,h2_molar_fraction=.05*scale))
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and float(auto)!=0.
    for eps in (1e-4,1e-5):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-10)
