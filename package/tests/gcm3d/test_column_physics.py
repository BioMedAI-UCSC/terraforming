import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY
from src.framework.physics.co2_condensation import simple_condensation
from src.framework.physics.column_physics import column_physics
from src.framework.physics.soil_frost import soil_layer_thickness

jax.config.update('jax_enable_x64',True)


def test_simple_condensation_latent_heat_matches_mass_away_from_guard():
    half=jnp.array([0.,50.,150.,350.,610.])
    full=(half[1:]+half[:-1])*.5
    temperature=jnp.array([100.,120.,140.,170.])
    t,mass,precip,guard=simple_condensation(temperature,half,full,BODY.cp_j_kg_k,BODY.gravity_m_s2)
    np.testing.assert_allclose(mass*590000.,(t-temperature)*BODY.cp_j_kg_k*
        jnp.diff(half)/BODY.gravity_m_s2,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(jnp.sum(mass),precip,rtol=2e-14)
    assert not np.any(guard)
    assert np.any(np.asarray(mass)>0.)


def test_original_mass_guard_is_exposed_instead_of_hiding_energy_mismatch():
    t,mass,_,guard=simple_condensation(jnp.array([-1000.]),jnp.array([0.,610.]),
        jnp.array([305.]),BODY.cp_j_kg_k,BODY.gravity_m_s2)
    assert bool(guard[0])
    assert float(mass[0])==61.
    assert float(t[0])>0.


def inputs():
    half=jnp.array([0.,50.,150.,350.,610.])
    return dict(temperature=jnp.array([110.,130.,145.,150.]),
        u=jnp.array([30.,20.,10.,5.]),v=jnp.array([-5.,-3.,-2.,-1.]),
        gas_tracers=jnp.arange(8,dtype=jnp.float64).reshape(4,2)*1e-5,
        soil=jnp.full(16,148.),pressure_half=half,pressure_full=(half[1:]+half[:-1])*.5,
        co2_ice_kg_m2=100.,radiative_heating_k_s=jnp.full(4,1e-5),
        downward_surface_flux_w_m2=0.,dt_seconds=300.,body=BODY,
        soil_thickness=soil_layer_thickness(16),soil_diffusivity=(250./(1500.*627.9))**2,
        soil_volumetric_heat_capacity=1500.*627.9,emissivity=.81,surface_exchange_start=0)


def test_coupled_column_reports_complete_mass_source_and_deposition():
    kwargs=inputs()
    result=jax.jit(lambda:column_physics(**kwargs))()
    np.testing.assert_allclose(result.co2_ice_kg_m2-kwargs['co2_ice_kg_m2'],
        jnp.sum(result.gas_mass_loss_kg_m2),rtol=2e-12,atol=2e-12)
    assert result.diagnostics['atmospheric_co2_precipitation_kg_m2']>0.
    assert not np.any(result.diagnostics['co2_mass_guard'])
    assert np.all(np.asarray(result.gas_tracers)>=0.)


def test_coupled_column_reverse_derivative_reaches_radiative_heating():
    kwargs=inputs()
    def objective(heating):
        result=column_physics(**{**kwargs,'radiative_heating_k_s':jnp.full(4,heating)})
        return jnp.sum(result.temperature)+jnp.sum(result.soil)+result.co2_ice_kg_m2
    auto=jax.grad(objective)(1e-5)
    assert np.isfinite(auto) and abs(float(auto))>0.
    for epsilon in (1e-7,3e-8,1e-8):
        finite=(objective(1e-5+epsilon)-objective(1e-5-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-5)


def test_coupled_column_rejects_invalid_surface_exchange_region():
    with pytest.raises(ValueError,match='layer index'):
        column_physics(**{**inputs(),'surface_exchange_start':4})
