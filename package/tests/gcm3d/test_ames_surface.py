import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.framework.physics.ames_surface import soil_layer_thickness,ice_depth,surface_step

jax.config.update('jax_enable_x64',True)


def advance(temperature,snow,down):
    return surface_step(jnp.full(16,temperature),soil_layer_thickness(16),
        (250./(1500.*627.9))**2,1500.*627.9,610.,snow,down,-2.,.5,.95,300.)


def test_original_grid_rules_and_invalid_grid():
    for n in (4,12,16,20,21):
        grid=soil_layer_thickness(n)
        assert len(grid)==n+1
        assert np.all(grid>0.)
    np.testing.assert_array_equal(soil_layer_thickness(12)[-1:],np.array([3.5]))
    np.testing.assert_array_equal(soil_layer_thickness(16)[12:15],np.array([3.,6.,12.]))
    np.testing.assert_allclose(soil_layer_thickness(21),.015*1.2**np.arange(22),rtol=0.)
    with pytest.raises(ValueError,match='two layers'):
        soil_layer_thickness(1)


def test_new_frost_and_exhausted_frost_use_reference_branches():
    soil,snow,d=advance(148.,0.,0.)
    assert bool(d['frost_retry'])
    assert float(snow)>0.
    np.testing.assert_allclose(soil[0],d['frost_temperature_k'],rtol=2e-14)
    assert float(soil[-1])==148.
    _,ice,_=advance(148.,.001,100.)
    np.testing.assert_allclose(ice,0.,atol=1e-12)


@pytest.mark.parametrize('temperature,snow,down',[(200.,0.,180.),(148.,0.,0.),(148.,100.,0.)])
def test_surface_and_phase_reverse_derivative(temperature,snow,down):
    def objective(flux):
        soil,ice,_=advance(temperature,snow,flux)
        return jnp.sum(soil)+ice
    auto=jax.grad(objective)(down)
    for eps in (1e-3,1e-4,1e-5):
        finite=(objective(down+eps)-objective(down-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-5,atol=1e-8)


def test_batched_independent_columns_match_single_columns():
    temperatures=jnp.array([148.,148.,200.])
    snow=jnp.array([0.,.001,0.])
    down=jnp.array([0.,100.,180.])
    soil=jnp.broadcast_to(temperatures,(16,3))
    result=surface_step(soil,soil_layer_thickness(16),jnp.full(15,(250./(1500.*627.9))**2),
        jnp.full(16,1500.*627.9),610.,snow,down,-2.,.5,.95,300.)
    for index in range(3):
        reference=advance(temperatures[index],snow[index],down[index])
        np.testing.assert_allclose(result[0][:,index],reference[0],rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(result[1][index],reference[1],rtol=2e-14,atol=2e-14)


def test_subsurface_ice_depth_has_north_south_symmetry_and_finite_gradient():
    north=ice_depth(.8,.1,60.)
    south=ice_depth(-.8,.1,-60.)
    np.testing.assert_allclose(north,south,rtol=2e-14)
    assert 0.<float(north)<=10.
    assert np.isfinite(jax.grad(lambda lat:ice_depth(lat,.1,60.))(.8))
