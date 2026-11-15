import numpy as np
import pytest
from scipy.optimize import brentq

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.water_bulk import (
    find_condensed_mass,saturation_ratio,phase_change,cloud_step,LATENT_HEAT)

jax.config.update('jax_enable_x64',True)


def test_water_saturation_root_and_implicit_derivative_match_independent_solve():
    p=610.;t=250.;q=.02;guess=-.05
    def exact(offset):
        residual=lambda x:q-x-float(saturation_ratio(t+offset+x*LATENT_HEAT/735.94,p))
        return brentq(residual,guess,0.,xtol=1e-15,rtol=1e-14)
    auto=jax.jit(jax.grad(lambda offset:find_condensed_mass(q,p,t+offset,guess)))(0.)
    np.testing.assert_allclose(find_condensed_mass(q,p,t,guess),exact(0.),rtol=3e-12,atol=1e-14)
    for eps in (1e-3,1e-4):
        finite=(exact(eps)-exact(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-7,atol=1e-10)


@pytest.mark.parametrize('regime',['clear','condense','evaporate','limited'])
def test_bulk_phase_conserves_water_and_latent_energy(regime):
    t=jnp.array([180.,200.,230.,260.]);p=jnp.array([30.,100.,300.,610.])
    sat=saturation_ratio(t,p)
    q=sat*(2. if regime=='condense' else .2)
    ice=jnp.full(4,.01 if regime=='evaporate' else 1e-7 if regime=='limited' else 0.)
    if regime=='clear': q=jnp.zeros(4)
    updated,vapor,cloud=jax.jit(phase_change)(t,p,q,ice)
    np.testing.assert_allclose(vapor+cloud,q+ice,rtol=3e-13,atol=1e-14)
    np.testing.assert_allclose(735.94*updated-LATENT_HEAT*cloud,
        735.94*t-LATENT_HEAT*ice,rtol=3e-13,atol=5e-10)
    assert np.min(vapor)>=0. and np.min(cloud)>=0.


@pytest.mark.parametrize('dt',[5.,300.,30000.])
def test_bulk_water_settling_and_precipitation_preserve_total_material(dt):
    half=jnp.array([0.,30.,100.,300.,610.]);full=.5*(half[:-1]+half[1:])
    t=jnp.array([160.,200.,240.,280.]);q=saturation_ratio(t,full)*2.
    ice=jnp.full(4,.002)
    result=jax.jit(cloud_step)(t,half,full,q,ice,260.,jnp.full(3,.001),dt)
    gas=jnp.diff(half)/3.72
    np.testing.assert_allclose(jnp.sum(gas*(q+ice)),
        jnp.sum(gas*(result.vapor_ratio+result.ice_ratio))+result.deposition_kg_m2,rtol=3e-13)
    np.testing.assert_allclose(result.diagnostics['rain_kg_m2']+result.diagnostics['snow_kg_m2'],
        result.deposition_kg_m2,rtol=2e-14)
    assert np.min(result.ice_ratio)>=0. and np.min(result.vapor_ratio)>=0.


def test_bulk_cloud_rollout_derivative_matches_actual_polished_forward():
    half=jnp.array([0.,30.,100.,300.,610.]);full=.5*(half[:-1]+half[1:])
    t=jnp.array([180.,210.,240.,260.]);q=jnp.array([.0001,.001,.01,.03]);ice=jnp.full(4,.002)
    def objective(offset):
        result=cloud_step(t+offset,half,full,q,ice,260.,jnp.full(3,.001),300.)
        return jnp.mean(result.temperature)+result.deposition_kg_m2
    auto=jax.jit(jax.grad(objective))(0.)
    assert np.isfinite(auto) and float(auto)!=0.
    for eps in (1e-3,1e-4):
        finite=(objective(eps)-objective(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-8)


def test_unbracketed_source_root_is_reported_as_invalid():
    value=find_condensed_mass(.01,610.,250.,0.)
    assert np.isnan(value)


def test_root_gradient_accepts_integer_pressure_and_material_constants():
    objective=lambda offset:find_condensed_mass(.02,610,250+offset,-.05,2800000,736)
    auto=jax.jit(jax.grad(objective))(0.)
    assert np.isfinite(auto)
    for eps in (1e-3,1e-4):
        np.testing.assert_allclose(auto,(objective(eps)-objective(-eps))/(2.*eps),rtol=3e-7,atol=1e-10)
