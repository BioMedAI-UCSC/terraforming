"""Learned components, conservative coupling, parameter gradients and restoration."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
from src.framework.gcm._dinosaur import jax, jnp, scales, primitive_equations
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm.learning import rollout, make_parameterized_step
from src.framework.physics.gcm import (
    initial_column_state, forced_primitive_equations, forced_co2_primitive_equations,
    two_stream_radiative_fluxes, surface_energy_tendencies,
)
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing, co2_forcing
from src.framework.neural import (
    NeuralRadiation, ColumnInputs, fit_normalization, column_features,
    inputs_from_state, reference_fluxes, radiation_budget_residual, heating_rates,
    feature_schema, save_checkpoint, load_checkpoint, directional_gradient_check,
)
from src.framework.neural.columns import initial_column, make_column_step, column_energy
from src.framework.neural.training import adam_init, adam_update, weighted_mse

jax.config.update("jax_enable_x64", True)
BOUNDARIES = (0., .5, 1.)


@pytest.fixture
def columns():
    return ColumnInputs(jnp.asarray([[[190.], [205.]], [[215.], [225.]]]),
                        jnp.asarray([[240.], [210.]]), jnp.asarray([[610.], [720.]]),
                        jnp.asarray(.2), jnp.asarray(.06), jnp.asarray(35.),
                        jnp.asarray([[500.], [0.]]), jnp.asarray([[1.2], [20.]]),
                        jnp.asarray(.25), jnp.asarray(.95))


@pytest.fixture
def forcing():
    return dataclasses.replace(radiative_forcing(), co2_radiation_enabled=True,
                               ames_correlated_k_enabled=False)


@pytest.fixture
def neural(columns, forcing):
    adapter = NeuralRadiation(BOUNDARIES, forcing.stefan_boltzmann, hidden_sizes=(8,), compact_shortwave=True)
    norm = fit_normalization(column_features(columns, BOUNDARIES), split="train")
    params = adapter.init(jax.random.PRNGKey(0))
    return adapter, params, norm


def assert_trees_equal(a, b, atol=1e-10):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=atol)


def test_rollout_samples_batch_and_rematerialization():
    step = lambda p, s: s + p*jnp.sin(s)
    initial, params = jnp.array([.1, .3]), .2
    expected = initial
    snapshots = []
    for i in range(5):
        expected = step(params, expected)
        if i in (1, 3, 4):
            snapshots.append(expected)
    result = jax.jit(lambda p, s: rollout(step, p, s, 5, 2))(params, initial)
    np.testing.assert_allclose(result.trajectory, jnp.stack(snapshots))
    np.testing.assert_array_equal(result.steps, [2, 4, 5])
    assert_trees_equal(result.final_state, expected)
    assert_trees_equal(result, rollout(step, params, initial, 5, 2, remat=True))
    assert rollout(step, params, initial, 5).trajectory is None
    sampled = rollout(step, params, initial, 2, 8)
    assert sampled.trajectory.shape == (1, 2)
    individual = jax.vmap(lambda s: rollout(step, params, s, 5).final_state)(initial)
    np.testing.assert_allclose(individual, expected)
    loss = lambda p, remat: jnp.sum(rollout(step, p, initial, 5, 2, remat=remat).final_state)
    assert_trees_equal(jax.grad(lambda p: loss(p, False))(params), jax.grad(lambda p: loss(p, True))(params))
    with pytest.raises(ValueError):
        rollout(step, params, initial, 0)
    with pytest.raises(ValueError):
        make_parameterized_step(lambda p: None, float("nan"), None)


def test_feature_normalization_and_checkpoint(tmp_path, columns, neural):
    adapter, params, norm = neural
    x = column_features(columns, BOUNDARIES)
    assert x.shape == (2, 1, 13)
    np.testing.assert_allclose(x[..., 2], .25*columns.surface_pressure_pa)
    np.testing.assert_allclose(norm.apply(x).mean(axis=(0, 1)), 0, atol=1e-12)
    with pytest.raises(ValueError, match="train"):
        fit_normalization(x, split="test")
    with pytest.raises(ValueError):
        fit_normalization(x, split="train", weights=jnp.asarray([[-1.], [1.]]))
    metadata = {"feature_schema": feature_schema(2), "physics": {"sigma": BOUNDARIES, "scheme": "compact"}}
    path = tmp_path / "model.npz"
    save_checkpoint(path, params, norm, adapter.model, metadata=metadata)
    restored, restored_norm = load_checkpoint(path, adapter.model, expected_metadata=metadata)
    assert_trees_equal(params, restored)
    assert_trees_equal(adapter.predict(params, columns, norm), adapter.predict(restored, columns, restored_norm))
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, adapter.model, expected_metadata={**metadata, "physics": {"scheme": "ames"}})
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="dtype"):
            load_checkpoint(path, adapter.model, expected_metadata=metadata)
    finally:
        jax.config.update("jax_enable_x64", True)
    with pytest.raises(ValueError, match="nonfinite"):
        save_checkpoint(path, params, norm._replace(mean=norm.mean*jnp.nan), adapter.model, metadata=metadata)


def test_boundary_conditions_and_energy_closure(columns, neural, forcing):
    adapter, params, norm = neural
    for flux in (adapter.predict(params, columns, norm), reference_fluxes(columns, BOUNDARIES, BODY, forcing)):
        for field in flux[:4]:
            assert np.all(np.asarray(field) >= 0)
        np.testing.assert_array_equal(flux.shortwave_down_w_m2[:, 1], 0)
        np.testing.assert_array_equal(flux.shortwave_up_w_m2[:, 1], 0)
        np.testing.assert_allclose(flux.shortwave_down_w_m2[0], columns.incoming_solar_w_m2)
        np.testing.assert_allclose(flux.shortwave_up_w_m2[-1], columns.albedo*flux.shortwave_down_w_m2[-1])
        np.testing.assert_array_equal(flux.longwave_down_w_m2[0], 0)
        np.testing.assert_allclose(flux.longwave_up_w_m2[-1],
                                   columns.emissivity*forcing.stefan_boltzmann*columns.surface_temperature_k**4)
        np.testing.assert_allclose(radiation_budget_residual(flux), 0, atol=1e-12)
        air, surface = heating_rates(flux, columns.surface_pressure_pa, BOUNDARIES, BODY, forcing.thermal_inertia)
        capacity = BODY.cp_j_kg_k*columns.surface_pressure_pa*.5/BODY.gravity_m_s2
        np.testing.assert_allclose(jnp.sum(air*capacity, axis=0)+surface*forcing.thermal_inertia,
                                   flux.toa_net_down_w_m2, atol=1e-12)


def test_neural_rollout_gradient_update_and_budget(columns, neural, forcing):
    adapter, params, norm = neural
    step = make_column_step(columns, BOUNDARIES, BODY, forcing.thermal_inertia,
                            lambda p, x: adapter.predict(p, x, norm), dt_seconds=5.)
    initial = initial_column(columns)
    loss = jax.jit(lambda p: jnp.mean((rollout(step, p, initial, 3, remat=True).final_state.air_temperature_k - 205)**2))
    direction = jax.tree.map(lambda p: jnp.ones_like(p)*.01, params)
    check = directional_gradient_check(loss, params, direction, epsilon=1e-4)
    assert abs(check["autodiff"]) > 1e-6
    assert check["relative_error"] < 1e-5
    updated, _ = adam_update(params, jax.grad(loss)(params), adam_init(params), learning_rate=1e-4)
    assert float(loss(updated)) < float(loss(params))
    final = rollout(step, params, initial, 5).final_state
    residual = (column_energy(final, columns, BOUNDARIES, BODY, forcing.thermal_inertia)
                - column_energy(initial, columns, BOUNDARIES, BODY, forcing.thermal_inertia)
                - final.external_energy_j_m2)
    np.testing.assert_allclose(residual, 0, atol=1e-7)


def test_parameter_recovery(columns, forcing):
    step = make_column_step(columns, BOUNDARIES, BODY, forcing.thermal_inertia,
                            lambda p, x: reference_fluxes(x, BOUNDARIES, BODY, forcing),
                            dt_seconds=30., exchange_fn=lambda p: 2*p)
    predict = jax.jit(lambda p: rollout(step, p, initial_column(columns), 12).final_state.surface_temperature_k)
    target = predict(1.6)
    loss = jax.jit(lambda p: weighted_mse(predict(p), target))
    check = directional_gradient_check(loss, jnp.asarray(.7), jnp.asarray(1.))
    assert check["relative_error"] < 1e-5
    p = jnp.asarray(.7)
    state = adam_init(p)
    @jax.jit
    def update(p, state):
        return adam_update(p, jax.grad(loss)(p), state, learning_rate=.05)
    for _ in range(120):
        p, state = update(p, state)
    assert abs(float(p)-1.6) < .02
    assert float(loss(p)) < float(loss(.7))*1e-3


@pytest.fixture(scope="module")
def global_state():
    coords = coordinate_system("T21", 2)
    specs = physics_specs(BODY)
    grid = coords.horizontal
    z = jnp.zeros((2,) + grid.modal_shape)
    ps = float(specs.nondimensionalize(610.*scales.units.pascal))
    dyn = primitive_equations.State(vorticity=z, divergence=z, temperature_variation=z,
                                    log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps))),
                                    sim_time=0.)
    return coords, specs, initial_column_state(dyn, coords, 220., specs)


@pytest.mark.parametrize("ames", [False, True])
def test_reference_adapter_is_identical_and_not_duplicated(global_state, forcing, ames):
    forcing = dataclasses.replace(forcing, ames_correlated_k_enabled=ames)
    coords, specs, initial = global_state
    expected = surface_energy_tendencies(initial, coords, specs, BODY, forcing)
    replaced = surface_energy_tendencies(initial, coords, specs, BODY, forcing,
                                         radiation_component=two_stream_radiative_fluxes)
    assert_trees_equal(expected, replaced)
    inputs = inputs_from_state(initial, coords, specs, BODY, forcing)
    assert_trees_equal(reference_fluxes(inputs, BOUNDARIES, BODY, forcing),
                       two_stream_radiative_fluxes(initial, coords, specs, BODY, forcing))
    for build in (lambda component: forced_primitive_equations(coords, BODY, forcing, specs=specs, radiation_component=component),
                  lambda component: forced_co2_primitive_equations(coords, BODY, forcing, co2_forcing(), specs=specs,
                                                                    radiation_component=component)):
        assert_trees_equal(build(None).explicit_terms(initial), build(two_stream_radiative_fluxes).explicit_terms(initial))
    with pytest.raises(ValueError, match="co2_radiation"):
        forced_primitive_equations(coords, BODY, dataclasses.replace(forcing, co2_radiation_enabled=False),
                                   radiation_component=two_stream_radiative_fluxes)


def test_custom_model_and_full_gcm_parameter_gradient(global_state, forcing):
    coords, specs, initial = global_state
    inputs = inputs_from_state(initial, coords, specs, BODY, forcing)
    adapter = NeuralRadiation(BOUNDARIES, forcing.stefan_boltzmann, hidden_sizes=(8,), compact_shortwave=True)
    norm = fit_normalization(column_features(inputs, BOUNDARIES), split="train")
    # A user-supplied one-parameter model exercises the public callable contract
    # and differentiation through an actual semi-implicit Dinosaur timestep.
    def apply(p, x):
        return jnp.ones(x.shape[:-1] + (12,))*p
    step = make_parameterized_step(
        lambda p: forced_primitive_equations(coords, BODY, forcing, specs=specs,
                                            radiation_component=adapter.bind(p, norm, apply=apply)),
        10., specs)
    loss = jax.jit(lambda p: jnp.mean(rollout(step, p, initial, 1).final_state.surface_temperature))
    check = directional_gradient_check(loss, jnp.asarray(.2), jnp.asarray(1.), epsilon=1e-3)
    assert abs(check["autodiff"]) > 1e-6
    assert check["relative_error"] < 1e-4
    final = jax.jit(lambda p: rollout(step, p, initial, 2).final_state)(.2)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))


def test_invalid_loss_weights_are_visible():
    assert np.isnan(weighted_mse(jnp.ones(2), jnp.zeros(2), jnp.zeros(2)))
    assert np.isnan(weighted_mse(jnp.ones(2), jnp.zeros(2), jnp.array([-1, 2])))
