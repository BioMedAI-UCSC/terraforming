import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.hydrostatic_initialization import surface_pressure,air_temperature
from src.framework.physics.vertical_grid import coefficients

jax.config.update('jax_enable_x64',True)


def test_native_pressure_correction_preserves_its_nonexact_mass_target():
    pressure=surface_pressure(jnp.zeros(5),jnp.arange(1.,6.))
    expected=564.+141.*np.log(564.)/np.log(705.)
    np.testing.assert_allclose(pressure,expected,rtol=2e-15)
    assert float(pressure[0])<705.


@pytest.mark.parametrize('isothermal',[False,True])
@pytest.mark.parametrize('temperature',[130.,170.,250.])
def test_cold_start_reverse_derivatives_match_finite_differences(isothermal,temperature):
    phi=jnp.linspace(-7000.,15000.,13)*3.72;weights=jnp.arange(1.,14.)
    ak,bk=coefficients(56,705.)
    def objective(scale):
        selected=temperature+7.*(scale-1.)
        ps=surface_pressure(phi*scale,weights,temperature_k=selected,top_pressure_pa=ak[0])
        half=ak[:,None]+bk[:,None]*ps
        t=air_temperature(half,ps,temperature_k=selected,isothermal=isothermal)
        return jnp.sum(ps*weights)/jnp.sum(weights)+jnp.mean(t)
    automatic=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(automatic) and abs(float(automatic))>1e-5
    # Keep both probes inside the source frost-floor branches.
    for epsilon in (1e-5,1e-6):
        finite=(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(automatic,finite,rtol=2e-5,atol=1e-7)


def test_nonisothermal_profile_uses_cumulative_pressure_without_top_and_frost_floor():
    half=jnp.array([2.,3.,7.,20.,610.]);ps=half[-1]
    temperature=air_temperature(half,ps,isothermal=False)
    cumulative=np.array([1.,5.,18.,608.])
    frost=3182.48/(23.3494-np.log(cumulative*.01))
    expected=np.maximum(frost,170.+(3.72/735.94)*11100.*np.log(cumulative/610.))
    np.testing.assert_allclose(temperature,expected,rtol=2e-15)
    assert np.any(np.asarray(temperature)==frost)
    with pytest.raises(ValueError,match='pressure interfaces'): air_temperature(jnp.ones(1),610.)
