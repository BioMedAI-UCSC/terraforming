import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.water_surface import surface_exchange

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('scheme',['bulk','moment','bin'])
@pytest.mark.parametrize('inject',[False,True])
@pytest.mark.parametrize('deposition',[False,True])
def test_water_surface_exchange_closes_material_budget_and_uses_source_layers(scheme,inject,deposition):
    half=jnp.array([0.,20.,100.,300.,610.]);q=jnp.full(4,.01 if deposition else 0.)
    frost=.0001 if not deposition else 0.
    result=surface_exchange(q,half,200.,frost,.01,1,300.,scheme=scheme,inject_through_pbl=inject)
    gas=jnp.diff(half)/3.72
    np.testing.assert_allclose(jnp.sum(gas*result.vapor_ratio)+result.surface_water_kg_m2,
        jnp.sum(gas*q)+frost,rtol=3e-13,atol=3e-14)
    assert float(result.surface_water_kg_m2)>=0.
    assert np.min(result.vapor_ratio)>=0.
    np.testing.assert_array_equal(result.vapor_ratio[:1 if inject else 3],q[:1 if inject else 3])
    assert np.sign(float(result.upward_flux_kg_m2_s))==(-1 if deposition else 1)


def test_liquid_surface_pressure_and_latent_diagnostics_have_reverse_gradients():
    half=jnp.array([0.,20.,100.,300.,610.]);q=jnp.full(4,.01)
    def objective(offset):
        result=surface_exchange(q,half,290.+offset,100.,.001,1,300.,
            liquid_vapor_pressure=True,bulk_latent_heat=True)
        return result.surface_water_kg_m2+result.latent_flux_w_m2+result.latent_derivative_w_m2_k
    auto=jax.jit(jax.grad(objective))(0.)
    assert np.isfinite(auto) and float(auto)!=0.
    for eps in (1e-3,1e-4):
        finite=(objective(eps)-objective(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-7,atol=1e-7)


def test_native_negative_frost_repair_is_explicit_for_moment_mode():
    result=surface_exchange(jnp.zeros(2),jnp.array([0.,100.,610.]),200.,-1.,.01,0,300.,scheme='moment')
    assert result.diagnostics['repaired_surface_mass_kg_m2']==1.
    assert result.surface_water_kg_m2==0.


def test_invalid_pbl_index_does_not_invent_an_exchange_layer():
    result=surface_exchange(jnp.zeros(2),jnp.array([0.,100.,610.]),200.,1.,.01,3,300.)
    assert np.isnan(result.vapor_ratio).all()


@pytest.mark.parametrize('liquid',[False,True])
def test_water_optics_keep_strict_frost_snow_and_temperature_boundaries(liquid):
    from src.framework.physics.water_surface import optical_properties
    snow=jnp.array([0.,0.,0.,0.,.01,2.01])
    water=jnp.array([2.,2.01,2.01,2.01,100.,100.])
    temp=jnp.array([200.,200.,272.999,273.,290.,290.])
    a,e=jax.jit(lambda:optical_properties(.25,.95,temp,snow,water,
        frost_threshold_m=.002,liquid_water=liquid))()
    np.testing.assert_allclose(a,[.25,.4,.4,.07 if liquid else .4,.25,.25])
    np.testing.assert_allclose(e,[.95,1.,1.,1.,.95,.95])


def test_bottom_only_exchange_ignores_unused_pbl_index():
    args=(jnp.zeros(2),jnp.array([0.,100.,610.]),200.,1.,.01)
    actual=surface_exchange(*args,-1,300.,inject_through_pbl=False)
    expected=surface_exchange(*args,1,300.,inject_through_pbl=False)
    np.testing.assert_array_equal(actual.vapor_ratio,expected.vapor_ratio)
    assert np.isfinite(actual.vapor_ratio).all()
