import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.solar_geometry import averaged_zenith,daylight_geometry,radiation_clock

jax.config.update('jax_enable_x64',True)


def test_fixed_calendar_preserves_elapsed_clock_and_day_fraction():
    elapsed=jnp.array([.371,3.371,668.371])*88704.
    calendar,sols,fraction=jax.jit(lambda value:radiation_clock(value,88704.,100))(elapsed)
    np.testing.assert_allclose(calendar,(100.+.371)*88704.,rtol=0.,atol=1e-8)
    np.testing.assert_allclose(sols,[.371,3.371,668.371],rtol=2e-15)
    np.testing.assert_allclose(fraction,.371,rtol=0.,atol=1e-13)
    np.testing.assert_array_equal(radiation_clock(elapsed,88704.)[0],elapsed)
    objective=lambda value:radiation_clock(value,88704.,100)[0]+3.*88704.*radiation_clock(value,88704.,100)[1]
    automatic=jax.grad(objective)(.371*88704.)
    np.testing.assert_allclose(automatic,4.,rtol=2e-15)
    for epsilon in (1.,.5):
        np.testing.assert_allclose(automatic,(objective(.371*88704.+epsilon)-objective(.371*88704.-epsilon))/(2.*epsilon),rtol=1e-8)


@pytest.mark.parametrize('fraction,half_angle',[(.25,.3),(.5,.03),(0.,.3),(.5,2.)])
def test_daylight_weighted_cosine_matches_direct_hour_angle_integral(fraction,half_angle):
    latitude=.6;declination=.3
    cosine,daylight=averaged_zenith(fraction,declination,0.,latitude,half_angle)
    center=fraction*2.*np.pi+np.pi
    hour=np.linspace(center-half_angle,center+half_angle,100001)
    incoming=np.maximum(np.sin(latitude)*np.sin(declination)+np.cos(latitude)*np.cos(declination)*np.cos(hour),0.)
    expected=np.trapezoid(incoming,hour)/(2.*half_angle)
    np.testing.assert_allclose(cosine*daylight,expected,rtol=2e-9,atol=3e-10)


def test_polar_day_and_night_have_finite_reference_limits():
    angle,cosine,fraction=daylight_geometry(.3,jnp.array([-jnp.pi*.5,jnp.pi*.5]))
    np.testing.assert_allclose(angle,[0.,np.pi])
    np.testing.assert_allclose(cosine,[0.,np.sin(.3)])
    np.testing.assert_array_equal(fraction,[0.,1.])


def test_solar_geometry_reverse_derivative_matches_finite_difference():
    def objective(scale):
        cosine,fraction=averaged_zenith(.41*scale,.3*scale,.2*scale,.6*scale,.03)
        _,mean,daylight=daylight_geometry(.3*scale,.6*scale)
        return cosine*fraction+mean*daylight
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto)
    for eps in (1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('constant',[589.,400.,800.])
def test_solar_constant_uses_mean_orbital_distance_and_retains_derivatives(constant):
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    from src.framework.physics.gcm import orbital_distance,solar_flux
    forcing=reference_radiative_forcing(solar_constant_w_m2=constant)
    time=123.*88704.
    np.testing.assert_allclose(solar_flux(time,forcing),
        constant*(forcing.semi_major_axis_m/orbital_distance(time,forcing))**2,rtol=2e-15)
    objective=lambda scale:solar_flux(time*scale,forcing)
    automatic=jax.grad(objective)(1.)
    assert np.isfinite(automatic) and automatic!=0.
    for epsilon in (1e-5,1e-6):
        np.testing.assert_allclose(automatic,(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon),rtol=2e-7)


@pytest.mark.parametrize('constant',[0.,-1.,np.nan,np.inf])
def test_solar_constant_rejects_invalid_values(constant):
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    with pytest.raises(ValueError,match='finite and positive'):
        reference_radiative_forcing(solar_constant_w_m2=constant)
