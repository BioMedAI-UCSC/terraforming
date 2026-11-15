"""Independent extensive budgets for phase change, settling, transport/restart."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.framework.physics import co2_clouds as clouds
from src.framework.physics import gcm as physics
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm.dynamics import stepper, integrate, reference_temperature
from src.framework.gcm.restart import save_restart, load_restart
from src.framework.gcm._dinosaur import primitive_equations, scales

jax.config.update("jax_enable_x64", True)


def inputs():
    temperature = jnp.array([125., 135., 145., 155.])[:, None, None]
    pressure = jnp.full((1, 1), 610.)
    ice = jnp.array([.01, .005, .002, .001])[:, None, None]
    wind = jnp.stack([jnp.arange(4.)[:, None, None] * 10., jnp.ones_like(temperature) * 5.])
    sigma = np.array([.125, .375, .625, .875])
    return temperature, pressure, ice, wind, sigma, np.full(4, .25), co2_forcing()


@pytest.mark.parametrize("deficit", [-5., 0., 10.])
def test_phase_exchange_closes_mass_energy_and_momentum(deficit):
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    rates = clouds.phase_exchange(t, ps, q, wind, t + deficit, sigma, dsigma, BODY, cf)
    def budgets(epsilon):
        return clouds.column_budgets(t + epsilon * rates[0], ps + epsilon * rates[1],
            q + epsilon * rates[2], wind + epsilon * rates[3], sigma, dsigma, BODY, cf)
    # Differentiate extensive budgets independently of the tendency construction.
    tangent = jax.jvp(budgets, (jnp.asarray(0.),), (jnp.asarray(1.),))[1]
    for value in tangent:
        np.testing.assert_allclose(value, 0., atol=2.e-10)
    assert np.all(np.sign(np.asarray(rates[1])) == -np.sign(deficit))
    assert all(np.isfinite(x).all() for x in rates)


@pytest.mark.parametrize("dt", [30., 300., 30000.])
def test_settling_closes_full_resolved_exchange_budgets_at_large_dt(dt):
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    before = clouds.column_budgets(t, ps, q, wind, sigma, dsigma, BODY, cf)
    new_t, new_q, new_wind, deposit, heat, impulse = clouds.settle(
        t, ps, q, wind, jnp.full_like(ps, 150.), sigma, dsigma, BODY, cf, dt)
    after = clouds.column_budgets(new_t, ps, new_q, new_wind, sigma, dsigma, BODY, cf)
    frost_enthalpy = cf.cloud_ice_heat_capacity_j_kg_k * 150. - cf.latent_heat_j_kg
    np.testing.assert_allclose(after[0] + deposit, before[0], rtol=2.e-14)
    np.testing.assert_allclose(after[1] + heat + deposit * frost_enthalpy, before[1], rtol=2.e-14)
    np.testing.assert_allclose(after[2] + impulse, before[2], rtol=2.e-14)
    assert np.all(np.asarray(new_q) >= 0.) and np.all(np.asarray(deposit) > 0.)
    assert np.all(np.asarray(new_t) > 0.)


def test_warm_air_sublimates_only_available_cloud():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    result = clouds.phase_exchange(t, ps, q, wind, t - 10000., sigma, dsigma, BODY, cf)
    gas = ps[None] * jnp.asarray(dsigma)[:, None, None] / BODY.gravity_m_s2
    expected = BODY.gravity_m_s2 * jnp.sum(gas * q, axis=0) / cf.exchange_timestep_s
    np.testing.assert_allclose(result[1], expected, rtol=2.e-14)
    zero = clouds.phase_exchange(t, ps, jnp.zeros_like(q), wind, t - 10., sigma, dsigma, BODY, cf)
    for rate in zero:
        np.testing.assert_allclose(rate, 0., atol=1.e-14)


def model():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(BODY)
    grid = coords.horizontal
    zeros = jnp.zeros((4,) + grid.modal_shape)
    ps_nd = float(specs.nondimensionalize(610. * scales.units.pascal))
    dyn = primitive_equations.State(vorticity=zeros, divergence=zeros, temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps_nd))), sim_time=0.)
    state = physics.initial_co2_state(dyn, coords, specs=specs, body=BODY)
    return coords, specs, state


def test_suspended_ice_survives_gas_projection_and_restart(tmp_path):
    coords, specs, initial = model()
    grid = coords.horizontal
    initial = initial._replace(dynamics=dataclasses.replace(initial.dynamics,
        tracers={clouds.CO2_CLOUD_TRACER: grid.to_modal(jnp.full((4,) + grid.nodal_shape, .01))}))
    before = physics.cloud_pressure_equivalent(initial, coords)
    step = physics.positivity_preserving_co2_step(lambda s: s._replace(co2_ice=s.co2_ice + .01 * jnp.exp(
        grid.to_nodal(s.dynamics.log_surface_pressure))), coords, specs)
    final = jax.jit(step)(initial)
    np.testing.assert_allclose(physics.cloud_pressure_equivalent(final, coords), before, rtol=2.e-12)
    final = final._replace(surface_momentum=final.surface_momentum + 2.)
    restored = load_restart(save_restart(final, tmp_path / "clouds.npz"))
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)


def test_checkpoint_diagnostics_include_suspended_co2():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "scripts/run_gcm3d_ablation.py"
    spec = importlib.util.spec_from_file_location("ablation_cloud_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coords, specs, state = model()
    grid = coords.horizontal
    state = state._replace(dynamics=dataclasses.replace(state.dynamics,
        tracers={clouds.CO2_CLOUD_TRACER: grid.to_modal(jnp.full((4,) + grid.nodal_shape, .01))}))
    report = module._checkpoint_diagnostics(state, coords, specs, 300.)
    assert report["mean_co2_cloud_ice_pa"] == pytest.approx(6.1, rel=1.e-12)
    assert report["total_co2_mass_kg"] == pytest.approx(
        report["atmosphere_plus_surface_reservoir_mass_kg"] * 1.01, rel=1.e-12)


def test_coupled_clouds_are_suspended_before_settling_and_transfer_ground_impulse():
    coords, specs, state = model()
    grid = coords.horizontal
    reference = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    temp = jnp.full((4,) + grid.nodal_shape, 125.)
    state = state._replace(dynamics=dataclasses.replace(state.dynamics, temperature_variation=grid.to_modal(temp - reference)))
    cf = co2_forcing()
    forcing = dataclasses.replace(radiative_forcing(diurnal=False), sensible_heat_transfer_w_m2_k=0.)
    equation = physics.forced_co2_primitive_equations(coords, BODY, forcing, cf, specs=specs)
    tendency = equation.explicit_terms(state)
    assert float(jnp.max(grid.to_nodal(tendency.dynamics.tracers[clouds.CO2_CLOUD_TRACER]))) > 0.
    # An atmosphere-to-cloud transfer does not deposit that mass immediately.
    dry_cf = dataclasses.replace(cf, atmospheric_condensation=False)
    dry = physics.forced_co2_primitive_equations(coords, BODY, forcing, dry_cf, specs=specs).explicit_terms(state)
    np.testing.assert_array_equal(tendency.co2_ice, dry.co2_ice)
    step = physics.positivity_preserving_co2_step(stepper(equation, 300., specs), coords, specs,
        body=BODY, co2_forcing=cf, dt_seconds=300.)
    final = jax.jit(lambda s: integrate(step, s, 4))(state)
    assert float(jnp.max(physics.cloud_pressure_equivalent(final, coords))) > 0.
    assert float(jnp.max(final.co2_ice)) > 0.
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))
