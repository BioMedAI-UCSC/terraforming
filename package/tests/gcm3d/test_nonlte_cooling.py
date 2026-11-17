import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.nonlte_cooling import (cooling,interpolate_profile,
    solar_heating_factor,infrared_blend)

jax.config.update('jax_enable_x64',True)


def test_reference_interpolation_retains_last_valid_value_outside_table():
    x=jnp.array([-8.,-5.,-2.]);y=jnp.array([1.,10.,2.])
    q=jnp.array([-10.,-7.,0.,-4.,-2.,4.])
    np.testing.assert_allclose(interpolate_profile(q,x,y),[1.,4.,4.,22./3.,2.,2.])


def test_cooling_has_accurate_temperature_and_composition_gradients():
    p=jnp.geomspace(1e-6,1.,12)
    def objective(offset):
        return jnp.sum(cooling(jnp.linspace(140.,210.,12)+offset,p,.9,.01+offset*1e-4))
    auto=jax.jit(jax.grad(objective))(0.)
    assert np.isfinite(auto) and abs(float(auto))>1e-10
    for eps in (1e-3,1e-4):
        finite=(objective(eps)-objective(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-7,atol=1e-12)


def test_radiation_ghost_queries_keep_original_interpolation_history():
    # Earlier ghost values affect a later above-table query in the source.
    ref=dict(pressure_mbar=jnp.array([.001,.01,.1]),temperature_k=jnp.array([100.,200.,300.]),
        co2_m3=jnp.ones(3)*1e12,o_m3=jnp.ones(3)*1e10,cooling_k_s=jnp.array([-.001,-.002,-.003]))
    grid=jnp.array([.1,1.,10.,100.,100.,200.,200.])
    p=jnp.array([100.,200.]);t=jnp.ones(2)*300.
    actual=cooling(t,p,.9,.01,reference=ref,radiation_pressure_pa=grid)
    direct=cooling(t,p,.9,.01,reference=ref)
    assert np.isfinite(actual).all()
    assert not np.allclose(actual,direct)


def test_heating_factors_use_source_units_and_distinct_lookup_pressures():
    np.testing.assert_allclose(solar_heating_factor(jnp.array([0.,.1,100.])),
        jnp.array([0.,22./23.,22000./22001.]))
    np.testing.assert_allclose(infrared_blend(2.,-1.,jnp.array([0.,.1])),[-1.,.5])
