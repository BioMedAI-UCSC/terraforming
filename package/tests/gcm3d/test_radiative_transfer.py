import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.radiative_transfer import column_fluxes

jax.config.update('jax_enable_x64',True)


def fluxes(temperature=200.03,pressure=613.,mu=.6):
    half=jnp.array([0.,.13,.51,1.])*pressure
    return column_fluxes(jnp.full(3,temperature),temperature,half,
        (half[1:]+half[:-1])*.5,mu,jnp.array([12.7,24.2,54.6,145.9,354.9,657.5,106.3])/1.524**2,
        .25,.95,3.72)


@pytest.mark.parametrize('mu',[.6,0.,-0.2])
def test_full_radiation_flux_budget_and_night(mu):
    result=jax.jit(lambda:fluxes(mu=mu))()
    assert all(np.isfinite(np.asarray(x)).all() for x in result)
    np.testing.assert_allclose(jnp.sum(result.atmospheric_convergence_w_m2)+result.surface_net_w_m2,
        result.toa_net_down_w_m2,rtol=2e-13,atol=1e-10)
    if mu<=0.:
        np.testing.assert_array_equal(result.shortwave_down_w_m2,0.)
        np.testing.assert_array_equal(result.shortwave_up_w_m2,0.)


def test_full_radiation_reverse_derivative_reaches_temperature_and_pressure():
    def objective(scale):
        result=fluxes(temperature=203.03*scale,pressure=613.*scale)
        return result.surface_net_w_m2+jnp.sum(result.atmospheric_convergence_w_m2*jnp.arange(1.,4.))
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for eps in (1e-6,1e-7):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-4,atol=1e-4)
