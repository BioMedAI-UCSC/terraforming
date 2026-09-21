"""Test experiment validity/accounting using transparent analytic oracles."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mars_calibration import paper


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "paper-smoke.json").read_text())


@pytest.mark.parametrize("key,value", [
    ("dt_seconds", 600), ("train_seconds", [300, 601]),
    ("validation_seconds", [600, 900]), ("truth", [0, 1, 1]),
    ("starts", []), ("active_sets", [[1, 1]]), ("max_evaluations", 1),
    ("gradient_absolute_tolerance", float("nan")),
])
def test_reject_invalid_protocol(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        paper.validate_config(config)


def test_shipped_protocols_are_valid(config):
    paper.validate_config(config)
    production = json.loads((Path(__file__).resolve().parents[1] / "paper-experiment.json").read_text())
    paper.validate_config(production)


@pytest.mark.parametrize("method", ["L-BFGS-B", "Powell"])
def test_optimizer_respects_budget_and_records_real_calls(method):
    seen = []
    def forward(x):
        return np.sum((x - 0.65)**2)
    def vg(x):
        return forward(x), 2 * (x - 0.65)
    result = paper.optimize(forward, vg, np.array([0.2, 0.3, 0.4]), method, 4, seen.append)
    assert 1 <= result["evaluations"] <= 4
    assert len(seen) == result["evaluations"]
    assert result["best"]["loss"] <= seen[0]["loss"]
    assert all(0 <= p <= 1 for r in seen for p in r["normalized_parameters"])
    assert all(r["oracle_seconds"] >= 0 for r in seen)


def test_gradients_near_zero_use_absolute_and_relative_criterion():
    objective = lambda x: np.sum(x**2)
    assert all(c["pass"] for c in paper.gradient_checks(objective, np.zeros(2), np.zeros(2), 1e-4, 1e-10))
    assert not all(c["pass"] for c in paper.gradient_checks(objective, np.ones(2), np.zeros(2), 1e-4, 1e-10))


def test_paired_ablation_controls(config):
    cases = paper.ablation_cases(config["truth"], 300)
    assert len(cases) == 13
    assert len({c["name"] for c in cases}) == 13
    assert next(c for c in cases if c["name"] == "half_timestep")["dt"] == 150
    assert next(c for c in cases if c["name"] == "no_co2_exchange")["co2_exchange"] is False
    for c in cases[1:7]:
        assert c["parameters"] == config["truth"]
    assert config["truth"] == [0.6, 0.8, 1.4]


def test_step_factory_preserves_phase1_closure_intervals(monkeypatch):
    d = paper.d
    captured = []
    forcing = d.radiative_forcing(diurnal=True, co2_radiation_enabled=True)
    coords = d.coordinate_system("T21", 12)
    specs = d.physics_specs(d.MARS_BODY_3D)
    monkeypatch.setattr(d, "forced_co2_primitive_equations",
                        lambda coords, body, f, cf, **kw: captured.append((f, cf)))
    monkeypatch.setattr(d, "stepper", lambda *args: lambda state: state)
    for dt in (300, 150):
        d._build_step(coords, specs, forcing, dt, orography=np.zeros(coords.horizontal.modal_shape))
    for f, cf in captured:
        assert f is forcing
        assert f.pbl_implicit_timestep_s == forcing.pbl_implicit_timestep_s
        assert cf.exchange_timestep_s == d.co2_forcing(energy_limited=True).exchange_timestep_s


def test_external_speed_data_requires_provenance(tmp_path):
    assert all(r["status"] == "not_measured" for r in paper.external_timings(None))
    path = tmp_path / "timing.json"
    path.write_text(json.dumps([{"simulator": "example", "wall_seconds": 2}]))
    with pytest.raises(ValueError, match="source"):
        paper.external_timings(path)
    row = dict(simulator="example", source="published table 1", hardware="CPU X, 4 cores",
               resolution="T21/L12", physics="dry", precision="float64",
               timed_region="forward steps; no IO", simulated_seconds=600, wall_seconds=2, dt_seconds=300)
    path.write_text(json.dumps([row]))
    assert paper.external_timings(path)[0]["simulated_seconds_per_wall_second"] == 300


class AnalyticModel:
    """Each parameter controls an independent observable; no physical skill claim."""
    weights = np.ones((1, 1))

    def trajectory(self, seconds, dt):
        times = paper.d.jnp.asarray(seconds) / 300
        def run(parameters):
            return (times[:, None] * parameters[None, :])[:, :, None, None, None]
        return run

    loss = paper.Model.loss


def test_synthetic_recovery_and_excluded_continuation(config, tmp_path):
    paper.d.jax.config.update("jax_enable_x64", True)
    config = copy.deepcopy(config)
    config["max_evaluations"] = 100
    config["active_sets"] = [[2], [0, 1, 2]]
    emitted = []
    report = paper.recovery(AnalyticModel(), config, tmp_path, emitted.append)
    assert report["status"] == "complete"
    assert report["scientific_acceptance_pass"]
    assert len(report["results"]) == 4
    for item in report["results"]:
        assert item["validation_loss"] < item["initial_validation_loss"]
        assert max(item["parameter_relative_error"]) < 0.05
        if item["active"] == [2]:
            np.testing.assert_array_equal(item["initial_parameters"][:2], config["truth"][:2])
        assert item["finite_difference_forward_calls"] == 4 * len(item["active"])
    targets = np.load(tmp_path / "synthetic-targets.npz")
    expected = np.maximum(targets["fields"][:2].std(axis=(0, 2, 3, 4)), 1)
    np.testing.assert_allclose(targets["normalization"], expected)
    assert (tmp_path / "trajectory-errors.csv").exists()
    assert (tmp_path / "recovery.png").exists()


def test_flat_model_does_not_claim_recovery(config, tmp_path):
    class FlatModel(AnalyticModel):
        def trajectory(self, seconds, dt):
            return lambda parameters: paper.d.jnp.ones((len(seconds), 3, 1, 1, 1)) + 0 * parameters.sum()
    report = paper.recovery(FlatModel(), config, tmp_path, lambda _: None)
    assert not report["scientific_acceptance_pass"]
    assert all(not item["acceptance_pass"] for item in report["results"])
