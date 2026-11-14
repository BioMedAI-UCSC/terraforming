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
from src.framework.gcm._dinosaur import primitive_equations, scales, spherical_harmonic

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


@pytest.mark.parametrize("offset,cloud", [(-10., 0.), (-10., .01), (5., .02), ("mixed", .01)])
def test_saturation_adjustment_reaches_frost_point_and_closes_extensive_budgets(offset, cloud):
    sigma = (np.arange(12) + .5) / 12
    dsigma = np.full(12, 1. / 12)
    ps = jnp.full((1, 1), 610.)
    frost = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * ps[None])
    delta = jnp.tile(jnp.array([-12., 3., -4., 7.]), 3)[:, None, None] if offset == "mixed" else offset
    t = frost + delta
    q = jnp.full_like(t, cloud)
    wind = jnp.stack([jnp.linspace(-30., 100., 12)[:, None, None], jnp.full_like(t, 5.)])
    cf = co2_forcing()
    before = clouds.column_budgets(t, ps, q, wind, sigma, dsigma, BODY, cf)
    new = clouds.saturation_adjustment(t, ps, q, wind, sigma, dsigma, BODY, cf)
    after = clouds.column_budgets(*new, sigma, dsigma, BODY, cf)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_allclose(a, b, rtol=2.e-12, atol=1.e-10)
    temperature, pressure, ice, velocity = map(np.asarray, new)
    assert np.all(pressure >= .9 * np.asarray(ps) - 1e-10)
    assert np.min(ice) >= -1e-13
    final_frost = np.asarray(physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * pressure[None]))
    assert np.max(final_frost - temperature) < 1e-6
    assert np.max(np.where(ice > 1e-12, np.abs(temperature - final_frost), 0.)) < 1e-6


def test_saturation_adjustment_leaves_clear_warm_air_unchanged():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    q = jnp.zeros_like(q)
    t = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * ps[None]) + 10.
    result = clouds.saturation_adjustment(t, ps, q, wind, sigma, dsigma, BODY, cf)
    for actual, expected in zip(result, (t, ps, q, wind), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_saturation_adjustment_handles_distinct_columns_and_nonuniform_layers():
    boundaries = np.array([0., .02, .12, .4, 1.])
    sigma, dsigma = .5 * (boundaries[1:] + boundaries[:-1]), np.diff(boundaries)
    ps = jnp.asarray([[430., 510., 610.], [700., 760., 830.]])
    frost = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * ps[None])
    t = frost + jnp.asarray([-8., 3., -4., 5.])[:, None, None]
    q = jnp.full_like(t, .02)
    wind = jnp.stack([jnp.broadcast_to(jnp.arange(4.)[:, None, None] * 15., t.shape), jnp.ones_like(t) * 5.])
    cf = co2_forcing()
    before = clouds.column_budgets(t, ps, q, wind, sigma, dsigma, BODY, cf)
    new = clouds.saturation_adjustment(t, ps, q, wind, sigma, dsigma, BODY, cf)
    for a, b in zip(before, clouds.column_budgets(*new, sigma, dsigma, BODY, cf), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-10)
    final_frost = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * new[1][None])
    assert float(jnp.max(jnp.abs(new[0] - final_frost))) < 1e-6


def test_saturation_guard_limits_net_condensation_without_inventing_heat():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    t, q = jnp.full_like(t, 50.), jnp.zeros_like(q)
    before = clouds.column_budgets(t, ps, q, wind, sigma, dsigma, BODY, cf)
    new = clouds.saturation_adjustment(t, ps, q, wind, sigma, dsigma, BODY, cf)
    for a, b in zip(before, clouds.column_budgets(*new, sigma, dsigma, BODY, cf), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(new[1], .9 * ps, rtol=1e-12)
    assert float(jnp.min(new[0])) > 0.
    # The numerical mass-loss guard leaves a measured deficit, never a T clamp.
    frost = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * new[1][None])
    assert float(jnp.max(frost - new[0])) > 1.


def test_saturation_adjustment_reverse_gradient_matches_local_finite_differences():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    t = physics.co2_frost_point_k(jnp.asarray(sigma)[:, None, None] * ps[None]) - 8.
    def objective(offset):
        adjusted = clouds.saturation_adjustment(t + offset, ps, q, wind, sigma, dsigma, BODY, cf)
        return jnp.mean(adjusted[0]) + .05 * jnp.mean(adjusted[1])
    analytic = float(jax.grad(objective)(0.))
    for epsilon in (1e-4, 1e-5, 1e-6):
        numerical = float((objective(epsilon) - objective(-epsilon)) / (2 * epsilon))
        assert analytic == pytest.approx(numerical, rel=1e-4, abs=1e-8)


def test_warm_air_sublimates_only_available_cloud():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    result = clouds.phase_exchange(t, ps, q, wind, t - 10000., sigma, dsigma, BODY, cf)
    gas = ps[None] * jnp.asarray(dsigma)[:, None, None] / BODY.gravity_m_s2
    expected = BODY.gravity_m_s2 * jnp.sum(gas * q, axis=0) / cf.exchange_timestep_s
    np.testing.assert_allclose(result[1], expected, rtol=2.e-14)
    zero = clouds.phase_exchange(t, ps, jnp.zeros_like(q), wind, t - 10., sigma, dsigma, BODY, cf)
    for rate in zero:
        np.testing.assert_allclose(rate, 0., atol=1.e-14)


@pytest.mark.parametrize("deposit", [-.001, .001])
@pytest.mark.parametrize("cloud_ratio", [0., .01])
def test_surface_exchange_closes_gas_cloud_frost_heat_and_ground_momentum(deposit, cloud_ratio):
    sigma = (np.arange(12) + .5) / 12
    dsigma = np.full(12, 1. / 12)
    t = jnp.linspace(140., 200., 12)[:, None, None]
    ps = jnp.full((1, 1), 610.)
    q = jnp.full_like(t, cloud_ratio)
    wind = jnp.stack([jnp.linspace(-30., 100., 12)[:, None, None], jnp.full_like(t, 5.)])
    cf = co2_forcing()
    surface, frost = jnp.full_like(ps, 150.), jnp.full_like(ps, 20.)
    rate = jnp.full_like(ps, deposit)
    heat, ice, acceleration, ground_heat, impulse = clouds.surface_exchange(
        t, ps, q, wind, surface, frost, rate, sigma, dsigma, BODY, cf)
    def budgets(epsilon):
        mass, energy, momentum = clouds.column_budgets(t + epsilon * heat,
            ps - epsilon * BODY.gravity_m_s2 * rate, q + epsilon * ice,
            wind + epsilon * acceleration, sigma, dsigma, BODY, cf)
        new_surface, new_frost = surface + epsilon * ground_heat, frost + epsilon * rate
        energy = energy + cf.thermal_inertia * new_surface + new_frost * (
            cf.cloud_ice_heat_capacity_j_kg_k * new_surface - cf.latent_heat_j_kg)
        return mass + new_frost, energy, momentum + epsilon * impulse
    tangent = jax.jvp(budgets, (jnp.asarray(0.),), (jnp.asarray(1.),))[1]
    for value in tangent:
        np.testing.assert_allclose(value, 0., atol=2.e-10)
    # The upper gas must flow down during deposition, despite no upper source.
    # A varying profile exposes the missing sigma redistribution term.
    assert float(jnp.max(jnp.abs(heat[:8]))) > 0.
    if cloud_ratio:
        # The cloud column is transported, never removed with the gas sink.
        def cloud_mass(epsilon):
            gas = (ps - epsilon * BODY.gravity_m_s2 * rate)[None] * jnp.asarray(dsigma)[:, None, None] / BODY.gravity_m_s2
            return jnp.sum(gas * (q + epsilon * ice))
        assert float(jax.jvp(cloud_mass, (jnp.asarray(0.),), (jnp.asarray(1.),))[1]) == pytest.approx(0., abs=1e-14)


def test_surface_exchange_zero_rate_is_identity_and_rejects_invalid_layer_ratio():
    t, ps, q, wind, sigma, dsigma, cf = inputs()
    rates = clouds.surface_exchange(t, ps, q, wind, jnp.full_like(ps, 150.),
        jnp.ones_like(ps), jnp.zeros_like(ps), sigma, dsigma, BODY, cf)
    for rate in rates:
        np.testing.assert_array_equal(rate, 0.)
    coords, specs, state = model()
    for ratio in (0., 1.1, float("nan")):
        with pytest.raises(ValueError, match="surface_exchange_pressure_ratio"):
            physics.forced_co2_primitive_equations(coords, BODY, radiative_forcing(),
                dataclasses.replace(cf, surface_exchange_pressure_ratio=ratio), specs=specs)


def test_surface_transport_preserves_cloud_mass_during_gas_escape():
    coords, specs, state = model()
    grid = coords.horizontal
    q = jnp.full((4,) + grid.nodal_shape, .01)
    state = state._replace(dynamics=dataclasses.replace(state.dynamics,
        tracers={clouds.CO2_CLOUD_TRACER: grid.to_modal(q)}))
    cf = dataclasses.replace(co2_forcing(), escape_rate_kg_s=1.e10)
    ps = jnp.full(grid.nodal_shape, 610.)
    zeros = jnp.zeros_like(q)
    rates = physics._co2_surface_transport(state, zeros, ps, (zeros, zeros),
        jnp.zeros_like(state.co2_ice), coords, specs, BODY, cf)
    scale = 1. / float(specs.nondimensionalize(1. * scales.units.second))
    ratio_rate = grid.to_nodal(rates[1]) / scale
    pressure_rate = -cf.escape_rate_kg_s * cf.gravity_m_s2 / BODY.surface_area_m2
    np.testing.assert_allclose(ps[None] * ratio_rate + q * pressure_rate, 0., atol=1e-13)


def test_frost_heat_capacity_scales_radiation_and_soil_fluxes_with_phase_heating():
    coords, specs, state = model()
    pressure_scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    state = state._replace(co2_ice=jnp.full_like(state.co2_ice, 20. * BODY.gravity_m_s2 / pressure_scale),
        ground_temperature=state.ground_temperature + 10.)
    forcing = dataclasses.replace(radiative_forcing(diurnal=False), tsi_1au_w_m2=0.,
        sensible_heat_transfer_w_m2_k=0., regolith_enabled=True)
    cf = dataclasses.replace(co2_forcing(), atmospheric_condensation=False,
        energy_limited=False, exchange_rate_pa_s_per_k=0.)
    surface = physics.surface_energy_tendencies(state, coords, specs, BODY, forcing)[1]
    soil = physics.regolith_conduction_tendencies(state, specs, forcing)[0]
    expected = (surface + soil) * cf.thermal_inertia / (
        cf.thermal_inertia + 20. * cf.cloud_ice_heat_capacity_j_kg_k)
    actual = physics.forced_co2_primitive_equations(
        coords, BODY, forcing, cf, specs=specs).explicit_terms(state).surface_temperature
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


def test_default_saturation_uses_predicted_air_and_forms_cloud_before_next_step_settling():
    coords, specs, state = model()
    grid = coords.horizontal
    reference = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    state = state._replace(dynamics=dataclasses.replace(state.dynamics,
        temperature_variation=grid.to_modal(jnp.full((4,) + grid.nodal_shape, 125.) - reference)))
    cf = co2_forcing()
    adjusted = physics.adjust_co2_cloud_saturation(state, coords, specs, BODY, cf)
    air = grid.to_nodal(adjusted.dynamics.temperature_variation) + reference
    pressure_scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    pressure = jnp.exp(grid.to_nodal(adjusted.dynamics.log_surface_pressure)) * pressure_scale
    frost = physics.co2_frost_point_k(jnp.asarray(coords.vertical.centers)[:, None, None] * pressure)
    assert float(jnp.max(frost - air)) < 1e-5
    assert float(jnp.max(physics.cloud_pressure_equivalent(adjusted, coords))) > 0.
    np.testing.assert_array_equal(adjusted.co2_ice, state.co2_ice)
    step = physics.positivity_preserving_co2_step(lambda s: s, coords, specs,
        body=BODY, co2_forcing=cf, dt_seconds=300., return_diagnostics=True)
    first, diagnostics = jax.jit(step)(state)
    assert np.isfinite(float(diagnostics.saturation_adjusted_exchange_energy_j))
    assert float(diagnostics.saturation_adjusted_total_mass_kg) == pytest.approx(
        float(diagnostics.settled_total_mass_kg), rel=1e-10)
    np.testing.assert_array_equal(first.co2_ice, state.co2_ice)
    second, _ = jax.jit(step)(first)
    assert float(jnp.max(second.co2_ice)) > 0.


def test_finite_rate_ablation_forms_suspended_clouds_before_settling():
    coords, specs, state = model()
    grid = coords.horizontal
    reference = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    temp = jnp.full((4,) + grid.nodal_shape, 125.)
    state = state._replace(dynamics=dataclasses.replace(state.dynamics, temperature_variation=grid.to_modal(temp - reference)))
    cf = dataclasses.replace(co2_forcing(), atmospheric_saturation_adjustment=False)
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


def test_negligible_cloud_ice_does_not_perturb_flow_after_repeated_settling():
    coords, specs, state = model()
    grid = coords.horizontal
    shape = (4,) + grid.nodal_shape
    pattern = 1.2 * jnp.sin(grid.latitudes)[None] + .7 * jnp.cos(grid.longitudes)[:, None] * jnp.cos(grid.latitudes)[None]
    speed_scale = float(specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
    wind = jnp.broadcast_to(10. * jnp.cos(grid.latitudes)[None, None], shape)
    vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(grid, wind / speed_scale, jnp.zeros_like(wind))
    state = state._replace(dynamics=dataclasses.replace(state.dynamics, vorticity=vor, divergence=div,
        temperature_variation=grid.to_modal(jnp.broadcast_to(pattern[None], shape)),
        tracers={clouds.CO2_CLOUD_TRACER: grid.to_modal(jnp.full(shape, 1e-30))}))
    def tick(current, _):
        return physics.settle_co2_clouds(current, coords, specs, BODY, co2_forcing(), 300.), None
    final, _ = jax.jit(lambda s: jax.lax.scan(tick, s, None, length=100))(state)
    for key in ("vorticity", "divergence", "temperature_variation"):
        np.testing.assert_allclose(getattr(final.dynamics, key), getattr(state.dynamics, key), atol=1e-10, rtol=1e-10)
