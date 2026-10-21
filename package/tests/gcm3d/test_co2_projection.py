"""Independent inventory and restart regressions for the CO2 projection contract."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing
from src.celestials.planets.mars.maps import initial_rest_state
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.restart import load_restart, save_restart
from src.framework.gcm.specs import physics_specs
from src.framework.gcm._dinosaur import scales
from src.framework.physics.gcm import initial_co2_state, positivity_preserving_co2_step

jax.config.update("jax_enable_x64", True)
_u = scales.units


@pytest.fixture(scope="module")
def model():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(BODY)
    dyn = initial_rest_state(coords, specs, BODY, np.zeros(coords.horizontal.nodal_shape))
    dyn = dataclasses.replace(dyn, sim_time=jnp.asarray(0.0))
    state = initial_co2_state(dyn, coords, specs=specs, body=BODY)
    return coords, specs, state


def mass_kg(state, coords, specs):
    """Compute mass independently from SI pressure and Gaussian area weights."""
    grid = coords.horizontal
    pressure = specs.dimensionalize(
        jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure)) + state.co2_ice,
        _u.pascal,
    ).magnitude[0]
    weights = jnp.asarray(grid.quadrature_weights)
    return float(jnp.sum(pressure * weights) / jnp.sum(weights) * BODY.surface_area_m2 / BODY.gravity_m_s2)


def source_step(model, mass_change, dt=300.0, escape_kg=0.0, pressure_error_pa=0.0):
    coords, specs, _ = model
    grid = coords.horizontal
    pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    dt_nd = float(specs.nondimensionalize(dt * _u.second))
    weights = jnp.asarray(grid.quadrature_weights)
    # Nonuniform source; the reference global amount is independent of its mask.
    mask = jnp.broadcast_to(1.0 + 0.25 * jnp.cos(grid.longitudes[:, None]), grid.nodal_shape)
    mask = mask * jnp.sum(weights) / jnp.sum(mask * weights)

    def step(state):
        source_pa = mass_change(state, dt) * BODY.gravity_m_s2 / BODY.surface_area_m2
        loss_pa = escape_kg * BODY.gravity_m_s2 / BODY.surface_area_m2
        pressure_pa = pressure_scale * jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))
        pressure_pa = pressure_pa + source_pa * mask[None] - loss_pa + pressure_error_pa
        dyn = dataclasses.replace(
            state.dynamics,
            log_surface_pressure=grid.to_modal(jnp.log(pressure_pa / pressure_scale)),
            sim_time=state.dynamics.sim_time + dt_nd,
            temperature_variation=state.dynamics.temperature_variation + 1e-4,
        )
        return state._replace(dynamics=dyn, surface_temperature=state.surface_temperature + 0.25)

    return step


@pytest.mark.parametrize("source_kg", [1e12, -1e12])
def test_declared_source_or_sink_survives_projection_with_exact_escape(model, source_kg):
    coords, specs, initial = model
    change = lambda state, dt: jnp.asarray(source_kg)
    escape_kg = 3e11
    raw_step = source_step(model, change, escape_kg=escape_kg, pressure_error_pa=0.01)
    step = positivity_preserving_co2_step(
        raw_step, coords, specs, body=BODY,
        co2_forcing=co2_forcing(escape_rate_kg_s=escape_kg / 300.0),
        dt_seconds=300.0, external_co2_mass_change_kg=change, return_diagnostics=True,
    )
    raw = raw_step(initial)
    final, diagnostics = jax.jit(step)(initial)
    expected = mass_kg(initial, coords, specs) + source_kg - escape_kg
    assert mass_kg(final, coords, specs) == pytest.approx(expected, rel=2e-14)
    assert float(diagnostics.target_total_mass_kg) == pytest.approx(expected, rel=2e-14)
    assert float(diagnostics.corrected_total_mass_kg) == pytest.approx(expected, rel=2e-14)
    assert float(diagnostics.raw_total_mass_kg) == pytest.approx(mass_kg(raw, coords, specs), rel=2e-14)
    assert float(diagnostics.external_co2_mass_change_kg) == source_kg
    assert float(diagnostics.escape_mass_kg) == pytest.approx(escape_kg)
    assert bool(diagnostics.valid_inventory)
    assert float(diagnostics.atmospheric_rescale_kg) < 0.0
    np.testing.assert_array_equal(final.surface_temperature, raw.surface_temperature)
    np.testing.assert_array_equal(final.ground_temperature, raw.ground_temperature)
    np.testing.assert_array_equal(final.dynamics.temperature_variation, raw.dynamics.temperature_variation)


def test_undeclared_source_is_still_a_closed_budget(model):
    coords, specs, initial = model
    raw_step = source_step(model, lambda state, dt: 1e12)
    step = positivity_preserving_co2_step(raw_step, coords, specs)
    final = jax.jit(step)(initial)
    assert mass_kg(final, coords, specs) == pytest.approx(mass_kg(initial, coords, specs), rel=2e-14)


def test_zero_source_preserves_the_existing_closed_step(model):
    coords, specs, initial = model
    raw_step = source_step(model, lambda state, dt: 0.0)
    closed = positivity_preserving_co2_step(raw_step, coords, specs)(initial)
    open_zero = positivity_preserving_co2_step(
        raw_step, coords, specs, body=BODY, dt_seconds=300.0,
        external_co2_mass_change_kg=lambda state, dt: 0.0,
    )(initial)
    for left, right in zip(jax.tree.leaves(closed), jax.tree.leaves(open_zero)):
        np.testing.assert_array_equal(left, right)


def test_external_source_and_internal_frost_transfer_are_counted_separately(model):
    coords, specs, initial = model
    pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    change = lambda state, dt: 1e12
    add_source = source_step(model, change)

    def with_condensation(state):
        raw = add_source(state)
        ps_nd = jnp.exp(coords.horizontal.to_nodal(raw.dynamics.log_surface_pressure))
        dyn = dataclasses.replace(
            raw.dynamics,
            log_surface_pressure=coords.horizontal.to_modal(jnp.log(ps_nd - 5.0 / pressure_scale)),
        )
        return raw._replace(dynamics=dyn, co2_ice=raw.co2_ice + 5.0 / pressure_scale)

    audited = positivity_preserving_co2_step(
        with_condensation, coords, specs, body=BODY, dt_seconds=300.0,
        external_co2_mass_change_kg=change, return_diagnostics=True,
    )
    final, diagnostics = jax.jit(audited)(initial)
    gain = mass_kg(final, coords, specs) - mass_kg(initial, coords, specs)
    assert gain == pytest.approx(1e12, rel=1e-9)
    np.testing.assert_allclose(final.co2_ice, 5.0 / pressure_scale, rtol=1e-14)
    assert float(diagnostics.external_co2_mass_change_kg) == 1e12
    assert float(diagnostics.negative_frost_repair_kg) == 0.0


def test_negative_frost_repair_is_reported_as_internal_transfer(model):
    coords, specs, initial = model
    pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    raw = initial._replace(co2_ice=jnp.full_like(initial.co2_ice, -2.0 / pressure_scale))
    step = positivity_preserving_co2_step(
        lambda state: state, coords, specs, body=BODY, return_diagnostics=True,
    )
    final, diagnostics = jax.jit(step)(raw)
    expected_repair = 2.0 * BODY.surface_area_m2 / BODY.gravity_m_s2
    assert float(diagnostics.negative_frost_repair_kg) == pytest.approx(expected_repair)
    assert float(diagnostics.external_co2_mass_change_kg) == 0.0
    assert mass_kg(final, coords, specs) == pytest.approx(mass_kg(raw, coords, specs), rel=2e-14)
    assert np.min(np.asarray(final.co2_ice)) == 0.0


@pytest.mark.parametrize("declared", [-1e30, np.nan, np.inf])
def test_impossible_or_nonfinite_target_is_not_silently_replaced_by_positive_mass(model, declared):
    coords, specs, initial = model
    step = positivity_preserving_co2_step(
        lambda state: state, coords, specs, body=BODY, dt_seconds=300.0,
        external_co2_mass_change_kg=lambda state, dt: declared, return_diagnostics=True,
    )
    final, diagnostics = jax.jit(step)(initial)
    assert not bool(diagnostics.valid_inventory)
    assert not np.all(np.isfinite(np.asarray(final.dynamics.log_surface_pressure)))


@pytest.mark.parametrize("restart_step", [1, 2, 3])
@pytest.mark.parametrize("end_seconds", [200.0, 250.0])
def test_scheduled_source_and_accounting_match_across_restart(model, tmp_path, restart_step, end_seconds):
    coords, specs, initial = model
    time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))

    def change(state, dt):
        elapsed = state.dynamics.sim_time * time_scale
        return jnp.clip(end_seconds - elapsed, 0.0, dt) * 1e10

    audited = positivity_preserving_co2_step(
        source_step(model, change, dt=100.0), coords, specs, body=BODY, dt_seconds=100.0,
        external_co2_mass_change_kg=change, return_diagnostics=True,
    )

    def rollout(state, count):
        def scan_step(current, unused):
            next_state, diagnostics = audited(current)
            return next_state, diagnostics.external_co2_mass_change_kg
        return jax.lax.scan(scan_step, state, None, length=count)

    full, sources = jax.jit(lambda state: rollout(state, 5))(initial)
    first, first_sources = jax.jit(lambda state: rollout(state, restart_step))(initial)
    save_restart(first, tmp_path / "restart.npz")
    resumed, last_sources = jax.jit(lambda state: rollout(state, 5 - restart_step))(
        load_restart(tmp_path / "restart.npz")
    )
    for left, right in zip(jax.tree.leaves(full), jax.tree.leaves(resumed)):
        np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)
    expected_injection = end_seconds * 1e10
    assert float(jnp.sum(sources)) == pytest.approx(expected_injection)
    assert float(jnp.sum(first_sources) + jnp.sum(last_sources)) == pytest.approx(expected_injection)
    # Repeated spectral transforms round a ~2.4e16 kg background inventory;
    # bound the error relative to the much smaller injected dose, not that background.
    gained = mass_kg(full, coords, specs) - mass_kg(initial, coords, specs)
    assert gained == pytest.approx(expected_injection, rel=1e-9)


def test_source_projection_remains_differentiable(model):
    coords, specs, initial = model

    def loss(source_kg):
        change = lambda state, dt: source_kg
        step = positivity_preserving_co2_step(
            source_step(model, change), coords, specs, body=BODY,
            dt_seconds=300.0, external_co2_mass_change_kg=change,
        )
        final = step(initial)
        pressure = jnp.exp(coords.horizontal.to_nodal(final.dynamics.log_surface_pressure))[0]
        weights = jnp.asarray(coords.horizontal.quadrature_weights)
        mean_pressure = jnp.sum(pressure * weights) / jnp.sum(weights)
        return mean_pressure * float(specs.dimensionalize(1.0, _u.pascal).magnitude) * BODY.surface_area_m2 / BODY.gravity_m_s2

    assert float(jax.grad(loss)(1e12)) == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("kwargs,match", [
    ({"external_co2_mass_change_kg": lambda state, dt: 1.0}, "body and dt_seconds"),
    ({"body": BODY, "external_co2_mass_change_kg": lambda state, dt: 1.0}, "body and dt_seconds"),
    ({"return_diagnostics": True}, "body is required"),
    ({"body": BODY, "dt_seconds": 0.0, "external_co2_mass_change_kg": lambda state, dt: 1.0}, "dt_seconds"),
])
def test_missing_or_invalid_budget_inputs_are_rejected(model, kwargs, match):
    coords, specs, _ = model
    with pytest.raises(ValueError, match=match):
        positivity_preserving_co2_step(lambda state: state, coords, specs, **kwargs)


def test_spatial_budget_is_rejected_instead_of_broadcast_as_global_mass(model):
    coords, specs, initial = model
    step = positivity_preserving_co2_step(
        lambda state: state, coords, specs, body=BODY, dt_seconds=300.0,
        external_co2_mass_change_kg=lambda state, dt: jnp.ones(coords.horizontal.nodal_shape),
    )
    with pytest.raises(ValueError, match="scalar global mass"):
        step(initial)
