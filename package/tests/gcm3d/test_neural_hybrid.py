"""Architecture, observed initialization, and coupled hybrid gradient checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from neural_hybrid import HybridExperiment, examples, fit_statistics
from train_neural_hybrid import curriculum, stage_horizon, active_params
from src.framework.physics.neural_column import initialize, apply
from src.framework.gcm._dinosaur import jax, jnp
from src.framework.physics import gcm as physics
from src.celestials.planets.mars.gcm import co2_forcing
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_neural_pbl import synthetic, BODY
jax.config.update("jax_enable_x64", True)


def setup():
    e = HybridExperiment(spinup=1, layers=4, flat_terrain=True)
    ds = synthetic(e, 4)
    x = next(examples(e, ds, 1))
    stats = fit_statistics(e, [ds])
    p = initialize(jax.random.key(0), len(stats["mean"]), 4, width=8, blocks=1)
    return e, ds, x, stats, p


def test_architecture_identity_and_ablation():
    e, ds, x, stats, p = setup()
    features = e.features(x[0], x[1])
    assert features.shape[-1] == 8 * 4 + 9
    for name in p:
        np.testing.assert_array_equal(apply(p[name], features), 0.)
        out = e.correct_state(x[0], e.output(p[name], x[0], x[1], stats))
        for a, b in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(x[0])):
            np.testing.assert_array_equal(a, b)
    altered = jax.tree_util.tree_map(jnp.ones_like, p)
    for mode, disabled in (("tendency-only", ("encoder", "decoder")), ("adapters-only", ("physics",))):
        masked = active_params(altered, mode)
        for name in disabled:
            np.testing.assert_array_equal(apply(masked[name], features), 0.)


def test_soil_initialization_uses_only_history_and_current_atmosphere():
    e, ds, x, stats, p = setup()
    changed = ds.copy(deep=True)
    changed["temp"].values[0] += 80.
    changed["tsurf"].values[2:] += 100.
    other = next(examples(e, changed, 1))
    for a, b in zip(jax.tree_util.tree_leaves(x[0]), jax.tree_util.tree_leaves(other[0])):
        np.testing.assert_array_equal(a, b)
    current, _ = e.snapshot(ds, 1)
    np.testing.assert_array_equal(e.fields(x[0]), e.fields(current))
    np.testing.assert_array_equal(x[0].dynamics.sim_time, current.dynamics.sim_time)
    cold = ds.copy(deep=True); cold["tsurf"].values[0] -= 30.
    colder = next(examples(e, cold, 1))[0]
    assert np.mean(colder.ground_temperature) < np.mean(x[0].ground_temperature)
    np.testing.assert_array_equal(e.fields(colder), e.fields(current))
    assert np.min(colder.ground_temperature) >= 190. - 1e-10
    assert np.max(colder.ground_temperature) <= 220. + 1e-10


def test_curriculum_boundaries():
    stages = curriculum("3:10,6:20,12:30")
    assert [stage_horizon(stages, n) for n in (0,9,10,19,20,30)] == [3,3,6,6,12,12]
    for invalid in ("3:10,1:20", "3:20,6:10", "0:5", "a", "1:2:3"):
        with pytest.raises(Exception):
            curriculum(invalid)


def test_no_learned_pressure_or_reservoir_source():
    e, _, x, _, _ = setup()
    state, context = x[:2]
    shape = state.dynamics.temperature_variation.shape
    residual = tuple(jnp.ones(shape) * 1e-5 for _ in range(3))
    def tendency(extra):
        return physics.forced_co2_primitive_equations(e.coords, BODY, e.forcing(context), co2_forcing(), specs=e.specs,
                   orography=e.orography, neural_tendencies=extra).explicit_terms(state)
    a, b = tendency(None), tendency(residual)
    for name in ("surface_temperature", "ground_temperature", "co2_ice"):
        np.testing.assert_array_equal(getattr(a,name), getattr(b,name))
    np.testing.assert_array_equal(a.dynamics.log_surface_pressure, b.dynamics.log_surface_pressure)
    assert np.max(np.abs(np.asarray(a.dynamics.temperature_variation - b.dynamics.temperature_variation))) > 0


@pytest.mark.slow
def test_actual_coupled_gradient_identity_and_learning():
    e, _, x, stats, p = setup()
    e.steps, e.dt, e.refresh_steps = 1, 30., 1
    stats["tendency_scale"] = np.ones((3,4))
    def parameters(value):
        return dict(p, physics=dict(p["physics"], decode=dict(p["physics"]["decode"], b=jnp.concatenate([jnp.full(4,value),jnp.zeros(8)]))))
    physical = jax.jit(lambda: e.forecast(None, x, stats, 1, physical=True))()
    hybrid = jax.jit(lambda: e.forecast(p, x, stats, 1))()
    np.testing.assert_allclose(hybrid, physical, rtol=1e-12, atol=1e-12)
    teacher = jax.jit(lambda: e.forecast(parameters(.4), x, stats, 1))()
    example = (x[0], x[1], teacher, x[3])
    def objective(values):
        params = parameters(values[0])
        for name, value in zip(("encoder", "decoder"), values[1:]):
            params[name] = dict(p[name], decode=dict(p[name]["decode"],
                                b=jnp.concatenate([jnp.full(4, value), jnp.zeros(8)])))
        return e.loss(params, example, stats, 1, 0.)
    value_grad = jax.jit(jax.value_and_grad(objective))
    values = jnp.zeros(3)
    initial, gradient = value_grad(values)
    assert np.isfinite(initial) and np.all(np.abs(gradient) > 1e-8)
    eps = 1e-3
    finite_difference = jnp.stack([
        (value_grad(values.at[i].set(eps))[0] - value_grad(values.at[i].set(-eps))[0]) / (2*eps)
        for i in range(3)])
    np.testing.assert_allclose(gradient, finite_difference, rtol=.01, atol=1e-8)
    for _ in range(8):
        loss, gradient = value_grad(values)
        values = values.at[0].add(-5 * gradient[0])
    final, _ = value_grad(values)
    assert float(final) < float(initial) * .5
