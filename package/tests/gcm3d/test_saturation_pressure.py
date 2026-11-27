"""Saturation lookup domains, source caps and physical derivatives."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.saturation_pressure import vapor_pressure,specific_humidity,load_tables

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('model',['smithsonian','simple'])
@pytest.mark.parametrize('phase',['blend','liquid','ice-liquid'])
def test_pressure_derivative_matches_polynomial_and_finite_difference(model,phase):
    pressure=lambda t:vapor_pressure(t,model=model,phase=phase).pressure_pa
    supplied=vapor_pressure(256.531,model=model,phase=phase)
    auto=jax.jit(jax.grad(pressure))(256.531);eps=1e-5
    np.testing.assert_allclose(auto,supplied.derivative_pa_k,rtol=2e-15)
    np.testing.assert_allclose(auto,(pressure(256.531+eps)-pressure(256.531-eps))/(2*eps),rtol=1e-8)


def test_lookup_retains_native_integer_truncation_domain_and_rejects_invalid_pressure():
    minimum=float(load_tables()['smithsonian_minimum_k'])
    result=vapor_pressure(jnp.array([minimum-.151,minimum-.149,623.209,623.211,jnp.nan]))
    np.testing.assert_array_equal(result.valid,[False,True,True,False,False])
    assert np.isnan(np.asarray(result.pressure_pa)[[0,3,4]]).all()
    assert not specific_humidity(200.,0.).valid


def test_humidity_cap_explicit_derivative_and_exact_formula():
    epsilon=189.02/461.5;pressure=vapor_pressure(350.)
    capped=specific_humidity(350.,100.)
    assert capped.specific_humidity==epsilon
    assert capped.temperature_derivative>0.
    assert jax.grad(lambda t:specific_humidity(t,100.).specific_humidity)(350.)==0.
    exact=specific_humidity(350.,1e5,vapor_ratio=.2,use_exact_qs=True)
    np.testing.assert_allclose(exact.specific_humidity,(1.+(1./epsilon-1.)*.2)*epsilon*pressure.pressure_pa/1e5,rtol=2e-15)
    ordinary=specific_humidity(230.531,610.)
    auto=jax.grad(lambda t:specific_humidity(t,610.).specific_humidity)(230.531)
    np.testing.assert_allclose(auto,ordinary.temperature_derivative,rtol=2e-15)


def test_unknown_saturation_models_and_phases_are_rejected():
    with pytest.raises(ValueError,match='model'):
        vapor_pressure(200.,model='unknown')
    with pytest.raises(ValueError,match='phase'):
        vapor_pressure(200.,phase='unknown')
