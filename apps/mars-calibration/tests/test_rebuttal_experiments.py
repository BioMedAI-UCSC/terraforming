"""Independent checks for the averaging, terrain and Viking data boundaries."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyse_rebuttal_climate import interval_mean, terrain_residual, read_viking


def test_interval_mean_integrates_ramp_at_fractional_boundaries():
    times = np.array([0., 1., 3., 5.])
    values = np.stack([2*times+3, 4*times-1], axis=1)
    np.testing.assert_allclose(interval_mean(times, values, .5, 4.), [7.5, 8.])
    with pytest.raises(ValueError):
        interval_mean(times, values, -1., 3.)


def test_pressure_residual_removes_known_hydrostatic_terrain_signal():
    z = np.array([[0., 1000.], [2000., 3000.]])
    residual = terrain_residual(610*np.exp(-z/10000), z, np.ones_like(z))
    np.testing.assert_allclose(residual, 0., atol=1e-14)


def test_viking_fixed_width_units_and_quality_flags(tmp_path):
    path = tmp_path / "viking.dat"
    path.write_text(' VL1  1  97.039    0  7.624  7.534  0.255  0.0670 18.01  0  0  0 18  0 49  0.000000 .F \n'
                    ' VL1  1  97.196    1  7.629  7.534  0.169  0.0687  2.28 -9 -9 -9 -9 -9 -9  0.000030 BB \n')
    rows = read_viking(path)
    assert len(rows) == 1
    assert rows[0]["station"] == "VL1"
    assert rows[0]["pressure_pa"] == pytest.approx(762.9)


@pytest.mark.parametrize("steps", [1, 3, 7, 12])
def test_block_checkpointing_preserves_final_and_mean_gradients(steps):
    from run_rebuttal_experiments import control_statistics, d
    d.jax.config.update("jax_enable_x64", True)

    def values(control):
        final, mean = control_statistics(lambda state: state+control, lambda state: state,
            d.jnp.asarray(0.), steps, block_steps=3)
        return d.jnp.stack([final, mean])

    result = d.jax.jit(values)(.25)
    derivative = d.jax.jit(d.jax.jacrev(values))(.25)
    np.testing.assert_allclose(result, [steps*.25, (steps+1)*.25/2], rtol=1e-12)
    np.testing.assert_allclose(derivative, [steps, (steps+1)/2], rtol=1e-12)


def test_block_checkpointing_rejects_empty_trajectory():
    from run_rebuttal_experiments import control_statistics
    with pytest.raises(ValueError):
        control_statistics(lambda x: x, lambda x: x, 0., 0)


@pytest.mark.parametrize("failure", ["gradient", "candidate"])
def test_invalid_gradient_or_candidate_never_wins_and_baselines_run(tmp_path, monkeypatch, failure):
    pytest.importorskip("cma")
    import dataclasses
    import json
    from types import SimpleNamespace
    from typing import NamedTuple
    import run_rebuttal_experiments as experiment
    d = experiment.d
    d.jax.config.update("jax_enable_x64", True)

    @dataclasses.dataclass
    class Forcing:
        albedo: object
        orbital_period_s: float = 59356800.

    class State(NamedTuple):
        surface_temperature: object

    @d.jax.custom_jvp
    def bad_gradient(value):
        return value

    @bad_gradient.defjvp
    def derivative(primals, tangents):
        return primals[0], d.jnp.full_like(tangents[0], d.jnp.nan)

    output = tmp_path / "run"
    def setup(*args):
        output.mkdir()
        grid = SimpleNamespace(nodal_shape=(2, 2), quadrature_weights=d.jnp.ones((2, 2)),
                               latitudes=d.jnp.array([-.5, .5]), longitudes=d.jnp.array([0., 3.]))
        return SimpleNamespace(horizontal=grid), None, Forcing(d.jnp.full((2, 2), .25)), None, State(d.jnp.ones((1, 2, 2))*200), None

    def build(coords, specs, forcing, dt, **kwargs):
        def advance(state):
            albedo = bad_gradient(forcing.albedo) if failure == "gradient" else forcing.albedo
            next_temperature = state.surface_temperature + albedo[None]
            if failure == "candidate":
                next_temperature = d.jnp.where(d.jnp.max(albedo) > .295, d.jnp.nan, next_temperature)
            return State(next_temperature)
        return advance

    monkeypatch.setattr(experiment, "setup", setup)
    monkeypatch.setattr(d, "_build_step", build)
    args = SimpleNamespace(inputs=tmp_path, output=output, objective="mean", steps=3,
        evaluations=8, seed=0, audit_directions=1, audit_only=False,
        methods=["gradient", "cma_es", "uniform"] if failure == "gradient" else ["uniform"])
    with pytest.raises(SystemExit) as status:
        experiment.spatial(args)
    assert status.value.code == 2
    report = json.loads((output / "report.json").read_text())
    if failure == "gradient":
        assert report["status"] == "gradient_audit_failed"
        assert report["results"]["gradient"]["forward_calls"] == 0
        assert report["results"]["cma_es"]["forward_calls"] == 8
    else:
        assert report["status"] == "nonfinite_optimization"
        assert report["results"]["uniform"]["failed_evaluations"] == 1
        assert report["results"]["uniform"]["trace"][-1]["objective_k"] is None
        assert report["scientific_gradient_acceptance_pass"] is None
    assert report["results"]["uniform"]["forward_calls"] == 8
    assert np.isfinite(report["results"]["uniform"]["best_objective_k"])
