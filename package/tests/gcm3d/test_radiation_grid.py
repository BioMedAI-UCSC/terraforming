import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.radiation_grid import radiation_grid

jax.config.update('jax_enable_x64',True)


def test_reference_interleaving_interface_temperature_and_lookup_nodes():
    grid=radiation_grid(jnp.array([.01,1.,3.,6.]),jnp.array([.3,2.,4.5]),
        jnp.array([150.,180.,210.]),220.)
    np.testing.assert_allclose(grid.pressure,[5e-8,.005,.01,.3,1.,2.,3.,4.5,6.])
    expected=180.+(150.-180.)*np.log(1./2.)/np.log(.3/2.)
    assert float(grid.temperature[4])==pytest.approx(expected)
    assert float(grid.temperature[-1])==220.
    np.testing.assert_allclose(grid.lookup_pressure,[.005,.005,.01,.01,1.,1.,3.,3.,6.])
    np.testing.assert_allclose(grid.delta_pressure,[.99,2.,3.])
    np.testing.assert_allclose(grid.lookup_temperature[2:4],150.)


@pytest.mark.parametrize('top',[0.,.01])
def test_grid_pressure_and_temperature_reverse_derivatives(top):
    def objective(scale):
        grid=radiation_grid(jnp.array([top,1.,3.,6.])*scale,
            jnp.array([.3,2.,4.5])*scale,jnp.array([150.,180.,210.])*scale,220.*scale)
        return sum(jnp.sum(x) for x in grid)
    auto=jax.jit(jax.grad(objective))(1.)
    finite=(objective(1.00001)-objective(.99999))/2e-5
    np.testing.assert_allclose(auto,finite,rtol=1e-8)


def test_dinosaur_zero_top_adds_no_ghost_mass_and_invalid_pressures_are_rejected():
    grid=radiation_grid(jnp.array([0.,1.,3.]),jnp.array([.3,2.]),jnp.array([150.,180.]),200.)
    assert all(np.isfinite(np.asarray(x)).all() for x in grid)
    np.testing.assert_array_equal(grid.pressure[:3],0.)
    assert float(jnp.sum(grid.delta_pressure))==3.
    invalid=radiation_grid(jnp.array([-.01,1.,3.]),jnp.array([.3,2.]),jnp.array([150.,180.]),200.)
    assert all(np.isnan(np.asarray(x)).all() for x in invalid)
    with pytest.raises(ValueError,match='must match'):
        radiation_grid(jnp.ones(2),jnp.ones(2),jnp.ones(2),200.)
