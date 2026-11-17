import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.optical_depth import aggregate_optics,gas_subgrid
from src.framework.physics.radiation_grid import radiation_grid

jax.config.update('jax_enable_x64',True)


def test_reference_ir_floor_and_absorption_only_limit():
    gas=jnp.zeros((6,5,17));zero=jnp.zeros((6,5))
    result=aggregate_optics(gas,zero,zero,zero,infrared=True)
    np.testing.assert_array_equal(result.optical_depth,1e-9)
    np.testing.assert_array_equal(result.scattering_albedo,0.)
    np.testing.assert_array_equal(result.asymmetry,0.)
    np.testing.assert_array_equal(result.cumulative,0.)


def test_rayleigh_retains_original_09999_scattering_fraction_and_special_bottom():
    gas=jnp.zeros((6,7,17));zero=jnp.zeros((6,7));ray=jnp.ones((6,7))*.1
    result=aggregate_optics(gas,zero,zero,zero,ray)
    np.testing.assert_allclose(result.optical_depth[:,0,0],[.2,.2,.1])
    np.testing.assert_allclose(result.scattering_albedo,.9999)
    np.testing.assert_array_equal(result.asymmetry,0.)
    np.testing.assert_allclose(result.cumulative[:,0,0],np.arange(7)*.1)


@pytest.mark.parametrize('top',[0.,.13])
def test_subgrid_gas_and_particle_reverse_derivatives(top):
    def objective(scale):
        # Stay away from the source table knots, where derivatives are one-sided.
        grid=radiation_grid(jnp.array([top,113.,317.,613.])*scale,
            jnp.array([57.,211.,466.])*scale,jnp.array([173.2,207.3,233.4])*scale,240.)
        gas=gas_subgrid(grid)[1]
        ext=jnp.full((8,5),.05)*scale
        result=aggregate_optics(gas,ext,.3*ext,.1*ext,infrared=True)
        return sum(jnp.sum(x) for x in result)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for eps in (1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-5)


def test_bad_particle_shape_is_rejected():
    with pytest.raises(ValueError,match='must match'):
        aggregate_optics(jnp.zeros((6,5,17)),jnp.zeros((5,5)),jnp.zeros((6,5)),jnp.zeros((6,5)))
