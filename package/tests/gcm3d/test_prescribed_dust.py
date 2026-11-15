import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.prescribed_dust import layer_opacities,fixed_optics,seasonal_fields

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('pulse',[0.,.2])
def test_prescribed_column_normalization_retains_surface_pressure_and_latitude_scaling(pulse):
    half=jnp.geomspace(.01,400.,13);full=jnp.sqrt(half[:-1]*half[1:])
    tau=layer_opacities(half,full,.3,latitude_rad=.7,latitude_variation=True,pulse_fraction=pulse)
    np.testing.assert_allclose(jnp.sum(tau),.3*(.3+.7*jnp.cos(.7))*400./610.,rtol=2e-14)
    assert np.min(tau)>=0.


def test_fixed_optics_preserves_visible_reference_column_and_keeps_ghosts_clear():
    half=jnp.array([0.,100.,300.,610.]);full=.5*(half[:-1]+half[1:])
    p=jnp.zeros(9).at[2::2].set(half).at[3::2].set(full)
    tau=jnp.array([.01,.02,.03])
    result=jax.jit(fixed_optics)(tau,p)
    np.testing.assert_allclose(jnp.sum(result.visible[0][:,5]),jnp.sum(tau),rtol=2e-14)
    for values in (*result.visible,*result.infrared):
        np.testing.assert_array_equal(values[:2],0.)
        assert np.isfinite(values).all()


def test_profile_and_size_optics_have_reverse_derivatives_through_pulses_and_pressure():
    def objective(scale):
        half=jnp.geomspace(.01,610.,13)*scale;full=jnp.sqrt(half[:-1]*half[1:])
        tau=layer_opacities(half,full,.3*scale,pulse_fraction=.2*scale)
        p=jnp.zeros(27).at[2::2].set(half).at[3::2].set(full)
        result=fixed_optics(tau,p,effective_radius_m=1.5e-6*scale)
        return jnp.sum(result.infrared[0]*jnp.arange(1.,27.)[:,None])
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-5
    eps=1e-5
    np.testing.assert_allclose(auto,(objective(1.+eps)-objective(1.-eps))/(2.*eps),rtol=2e-7,atol=1e-8)


def test_seasonal_interpolation_wraps_with_the_supplied_360_degree_endpoint():
    axis=jnp.array([0.,90.,180.,270.,360.]);values=jnp.array([1.,2.,3.,4.,1.])
    for ls,expected in ((0.,1.),(45.,1.5),(315.,2.5),(360.,1.),(-45.,2.5)):
        actual=seasonal_fields(ls,axis,values,values*10.)
        np.testing.assert_allclose(actual,[expected,expected*10.],rtol=1e-15)


def test_missing_tracer_height_inputs_are_rejected_and_empty_source_height_stays_invalid():
    half=jnp.geomspace(.01,610.,9);full=jnp.sqrt(half[:-1]*half[1:])
    with pytest.raises(ValueError,match='requires an aerosol'):
        layer_opacities(half,full,.3,cycle_scheme=1)
    invalid=layer_opacities(half,full,.3,cycle_scheme=1,aerosol_ratio=jnp.zeros(8))
    assert np.isnan(invalid).all()
