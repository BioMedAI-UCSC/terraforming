import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.solar_geometry import averaged_zenith,daylight_geometry

jax.config.update('jax_enable_x64',True)


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
