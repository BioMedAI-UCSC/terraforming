"""Reference coagulation material regularization and derivative checks."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.dust_coagulation import coagulation_step

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('direct',[False,True])
@pytest.mark.parametrize('modes',[1,2])
def test_active_coagulation_retains_source_small_mass_regularization(direct,modes):
    mass=jnp.array([1e-5,2e-5])[:modes];number=jnp.array([1e8,1.5e8])[:modes]
    result=jax.jit(lambda m,n:coagulation_step(190.,610.,m,n,300.,direct=direct))(mass,number)
    assert bool(result.diagnostics['active'])
    assert np.all(mass+result.mass_tendency*300.>0.)
    assert np.all(number+result.number_tendency*300.>0.)
    assert abs(float(jnp.sum(result.mass_tendency)*300.))<2e-15
    assert np.sum(result.number_tendency)<0.


@pytest.mark.parametrize('direct',[False,True])
def test_coagulation_reverse_derivative_reaches_temperature_pressure_and_dust(direct):
    def objective(x):
        mass=jnp.array([1e-5,2e-5])*x[2];number=jnp.array([1e8,1.5e8])*x[2]
        return jnp.sum(coagulation_step(x[0],x[1],mass,number,300.,direct=direct).number_tendency)/1e5
    x=jnp.array([190.,610.,1.]);auto=jax.jit(jax.grad(objective))(x)
    assert np.all(np.isfinite(auto)) and np.all(np.abs(auto)>0.)
    fd=[]
    for i,eps in enumerate([1e-3,1e-2,1e-4]):
        fd.append((objective(x.at[i].add(eps))-objective(x.at[i].add(-eps)))/(2.*eps))
    np.testing.assert_allclose(auto,fd,rtol=1e-4,atol=1e-9)


def test_inactive_source_mass_loss_is_exposed_instead_of_claiming_conservation():
    mass=jnp.array([1e-12]);number=jnp.array([10.])
    result=coagulation_step(190.,610.,mass,number,300.)
    assert not bool(result.diagnostics['active'])
    assert float(result.diagnostics['dust_mass_change_kg_kg'])<0.
    np.testing.assert_allclose(mass+300.*result.mass_tendency,300.e-15,rtol=1e-13)
