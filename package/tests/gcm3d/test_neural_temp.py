"""Scientific contracts, leakage, optimization and failure/resume checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from neural_temp_data import SPLITS, select_manifest, digest, freeze, write_json
from neural_temp_model import (SCHEMA, NO_GCM, ResidualMLP, build_features,
    fit_normalization, fit_controls, standardize, predict_neural)
from cache_neural_temp import cache_entry, read_entry, spinup_soil, physical_forcing
from train_neural_temp import train_model, load_checkpoint, mse, verify_frozen
from evaluate_neural_temp import metrics, bootstrap, evaluate
from run_neural_temp import device_environments, supervise, remaining


def metadata():
    years = [y for v in SPLITS.values() for y in v]
    # Deliberately reorder complete blocks rather than sort the source.
    years = years[::2] + years[1::2]
    return (np.concatenate([np.arange(960)/12 + (y-24)*700 for y in years]),
            np.repeat(years, 960), np.tile(np.arange(960)/960*360, len(years)))


def test_manifest_unordered_complete_deterministic_disjoint():
    args = metadata()
    m = select_manifest(*args)
    assert m == select_manifest(*args)
    assert m["achieved"] == {"train": 112, "validation": 32, "test": 32}
    assert not m["deficits"]
    assert m["exclusions"]
    for split, years in SPLITS.items():
        for year in years:
            rows = [r for r in m["starts"] if r["year"] == year]
            assert len(rows) == 16
            assert all(r["split"] == split for r in rows)
            assert np.min(np.diff(sorted(r["start_sol"] for r in rows))) >= 2-1e-9
            assert len(set(i for r in rows for i in r["indices"])) == 14*16
    assert not any(r["year"] == 28 for r in m["starts"])


def test_manifest_gaps_nonfinite_and_deficit_logged():
    t, y, ls = metadata()
    t[100] = np.nan
    t[200:] += .02
    m = select_manifest(t, y, ls, validator=lambda r: (_ for _ in ()).throw(ValueError("invalid field")) if r["year"] == 24 else None)
    assert any(r["reason"] == "nonfinite_metadata" for r in m["exclusions"])
    assert any(r["reason"] == "native_cadence_gap_or_order" for r in m["exclusions"])
    assert len(m["deficits"]) == 16
    assert m["achieved"]["train"] == 96


def synthetic(n=16, cells=32, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, cells, len(SCHEMA)))
    x[..., 0] += 210
    x[..., 7] = x[..., 0] - 3
    area = np.broadcast_to(np.linspace(1, 3, cells), (n, cells)).copy()
    area /= area.sum(axis=-1, keepdims=True)
    return {"features": x, "target": x[..., 7]+3+2*np.tanh(x[..., 1]), "area": area,
            "pressure": np.full((n, cells), 610.), "day": x[..., 2] > 0,
            "lat": np.tile(np.linspace(-85, 85, cells), (n, 1)),
            "lon": np.tile(np.linspace(0, 350, cells), (n, 1)),
            "lst": np.tile(np.linspace(0, 24, cells), (n, 1))}


def test_feature_order_and_missing_values():
    fields = {n: np.full((2, 3), i) for i, (n, _) in enumerate(SCHEMA)}
    x = build_features(fields)
    np.testing.assert_array_equal(x[0, 0], np.arange(len(SCHEMA)))
    fields.pop("start_dust")
    with pytest.raises(ValueError, match="missing"):
        build_features(fields)
    fields["start_dust"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_features(fields)


def test_training_only_weighted_normalization_controls():
    x = np.zeros((2, 2, len(SCHEMA)))
    x[..., 0] = [[1, 3], [5, 7]]
    y = 2 + 4*x[..., 0]
    w = np.array([[1, 3], [3, 1]])
    norm = fit_normalization(x, y, np.zeros_like(y), w, "train")
    assert norm["mean"][0] == 4
    params = fit_controls(x, y, np.zeros_like(y), w, norm, "train")
    assert params["constant"] == 18
    z = standardize(x, norm)[..., 0]
    expected_slope = np.sum(w/w.sum(axis=1, keepdims=True)/2*z*y)/(1+.001)
    assert params["ridge"][1] == pytest.approx(expected_slope)
    assert params["ridge"][0] == pytest.approx(18)
    assert min(norm["scale"]) > 0
    with pytest.raises(ValueError):
        fit_normalization(x, y, y, w, "validation")
    with pytest.raises(ValueError):
        fit_controls(x, y, y, w, norm, "test")


def test_zero_identity_and_finite_gradients():
    model = ResidualMLP(24)
    x = torch.randn(8, 24, dtype=torch.float64)
    assert torch.equal(model(x), torch.zeros(8, dtype=torch.float64))
    (model(x)-1).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.net[-1].weight.grad.abs().sum() > 0


def test_learning_resume_equivalence_and_no_gcm(tmp_path):
    train, val = synthetic(), synthetic(8, seed=5)
    contract = {"synthetic_test": True}
    full = train_model(tmp_path / "full", "neural", train, val, contract, stop_after=20)
    train_model(tmp_path / "resume", "neural", train, val, contract, stop_after=7)
    resumed = train_model(tmp_path / "resume", "neural", train, val, contract, stop_after=20)
    for k in full["parameters"]:
        torch.testing.assert_close(full["parameters"][k], resumed["parameters"][k], rtol=0, atol=0)
    initial = mse(val["features"][..., 7], val["target"], val["area"])
    assert mse(predict_neural(full, val["features"]), val["target"], val["area"]) < .6*initial
    ablation = train_model(tmp_path / "ablation", "no_gcm", train, val, contract, stop_after=3)
    assert all(not SCHEMA[i][0].startswith("forecast_") for i in NO_GCM)
    changed = val["features"].copy()
    changed[..., 7:15] += 1000
    np.testing.assert_array_equal(predict_neural(ablation, changed), predict_neural(ablation, val["features"]))
    with pytest.raises(ValueError, match="resume contract"):
        train_model(tmp_path / "resume", "neural", train, val, {"changed": True})


def test_soil_constant_and_past_only():
    pytest.importorskip("dinosaur")
    forcing = physical_forcing()
    history = np.full((13, 2, 3), 200.)
    result = spinup_soil(history, forcing, forcing.rotation_period_s/12)
    np.testing.assert_allclose(result, 200, atol=1e-10)
    history[0] = 190
    result = spinup_soil(history, forcing, forcing.rotation_period_s/12)
    assert result.min() >= 190-1e-9 and result.max() <= 200+1e-9
    # Final sample is available at start; prescribed integration uses left boundaries.
    history[-1] = 500
    np.testing.assert_array_equal(result, spinup_soil(history, forcing, forcing.rotation_period_s/12))


def test_atomic_cache_contract_checksum_and_incomplete(tmp_path):
    record, contract = {"id": "a"}, {"version": 1}
    arrays = {k: v[0] for k, v in synthetic().items()}
    assert read_entry(tmp_path, record, contract) is None
    cache_entry(tmp_path, record, contract, arrays)
    result = read_entry(tmp_path, record, contract)
    np.testing.assert_array_equal(result["target"], arrays["target"])
    with np.load(tmp_path / "cache/a/features.npz") as data:
        assert "target" not in data.files and "physical_error" not in data.files
    with pytest.raises(ValueError, match="contract"):
        read_entry(tmp_path, record, {"version": 2})
    (tmp_path / "cache/a/target.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_entry(tmp_path, record, contract)


def test_weighted_metrics_hand_calculation_and_blocks():
    p = np.array([[1., 3.], [2., 4.]])
    y = np.zeros_like(p)
    w = np.array([[3., 1.], [1., 3.]])
    result = metrics(p, y, w)
    assert result["rmse"] == pytest.approx(np.sqrt(8))
    assert result["bias"] == pytest.approx(2.5)
    assert result["rmse"] != pytest.approx((np.sqrt(3)+np.sqrt(13))/2)
    assert metrics(p, y, w, np.array([[True], [False]]))["starts"] == 1
    a = bootstrap([2, 2, 4, 4], ["a", "a", "b", "b"])
    assert a["block_counts"] == [2, 2]
    assert a["ci95"] == [2, 4]
    assert a == bootstrap([2, 2, 4, 4], ["a", "a", "b", "b"])


def test_evaluation_artifacts_and_regressions(tmp_path):
    data = synthetic(8)
    records = [{"id": str(i), "split": "validation", "quadrant": i//2, "block": str(i//2)} for i in range(8)]
    predictions = {n: data["features"][..., 7].copy() for n in ("physical", "persistence", "constant", "linear", "neural", "no_gcm")}
    report = evaluate(tmp_path, "validation", records, data, predictions)
    assert not report["scientific_success"]
    assert not report["criteria"]["neural_value"]
    assert len(report["breakdowns"]["neural"]) == 11
    assert (tmp_path / "plots/validation.png").exists()
    assert (tmp_path / "validation_predictions.npz").exists()


def test_freeze_and_test_gate(tmp_path):
    freeze(tmp_path / "contract.json", {"x": 1})
    with pytest.raises(ValueError):
        freeze(tmp_path / "contract.json", {"x": 2})
    write_json(tmp_path / "frozen_models.json", {"contract_hash": digest({}), "files": {}, "validation_gate": False})
    with pytest.raises(ValueError, match="gate failed"):
        verify_frozen(tmp_path, {}, require_gate=True)


def test_routing_failure_timeout_and_budget(tmp_path):
    envs = device_environments(["3", "1", "7", "5"])
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["3", "1", "7", "5"]
    with pytest.raises(ValueError):
        device_environments(["0"]*4)
    with pytest.raises(RuntimeError, match="worker failed"):
        supervise([[sys.executable, "-c", "raise SystemExit(4)"]], [os.environ.copy()], [tmp_path / "failure.log"], 10)
    with pytest.raises(TimeoutError):
        supervise([[sys.executable, "-c", "import time; time.sleep(10)"]], [os.environ.copy()], [tmp_path / "timeout.log"], .2)
    assert remaining([{"stage": "train_validation", "charged_gpu_hours": 2}], "train_validation") == .25
    assert remaining([{"stage": "test", "charged_gpu_hours": 4}], "train_validation") == 0


def test_cache_shards_resume_and_worker_failure_accounting(tmp_path, monkeypatch):
    import run_neural_temp as runner
    records = [{"id": str(i)} for i in range(4)]
    contract, ledger = {}, []
    arrays = {k: v[0] for k, v in synthetic().items()}
    routed = []
    def fake_supervise(commands, environments, logs, timeout, progress, on_spawn):
        assert len(commands) == 4
        routed.extend(e["CUDA_VISIBLE_DEVICES"] for e in environments)
        on_spawn([100001, 100002, 100003, 100004])
        for record in records:
            cache_entry(tmp_path, record, contract, arrays)
        progress(.1)
        return .1
    monkeypatch.setattr(runner, "supervise", fake_supervise)
    runner.cache_stage(tmp_path, records, contract, ["0", "1", "2", "3"], "train_validation", ledger)
    assert routed == ["0", "1", "2", "3"]
    assert ledger[0]["status"] == "complete"
    runner.cache_stage(tmp_path, records, contract, ["0", "1", "2", "3"], "train_validation", ledger)
    assert len(ledger) == 1
    def failed(*a, **k):
        raise RuntimeError("simulated GPU failure")
    monkeypatch.setattr(runner, "supervise", failed)
    with pytest.raises(RuntimeError, match="GPU failure"):
        runner.cache_stage(tmp_path, [{"id": "missing"}], contract, ["0", "1", "2", "3"], "train_validation", ledger)
    assert ledger[-1]["status"] == "failed"
    assert json.loads((tmp_path / "budget.json").read_text()) == ledger


def test_hard_kill_reconciliation_preserves_charges(monkeypatch):
    from run_neural_temp import reconcile_reservations
    ledger = [{"status": "reserved", "workers": 1, "pids": [123], "charged_gpu_hours": .2}]
    def dead(*a):
        raise ProcessLookupError()
    monkeypatch.setattr(os, "kill", dead)
    reconcile_reservations(ledger)
    assert ledger[0]["status"] == "interrupted_full_charge"
    assert ledger[0]["charged_gpu_hours"] == .2
    ledger[0]["status"] = "reserved"
    monkeypatch.setattr(os, "kill", lambda *a: None)
    with pytest.raises(RuntimeError, match="still be alive"):
        reconcile_reservations(ledger)


def test_pilot_cost_refuses_campaign(tmp_path, monkeypatch):
    import run_neural_temp as runner
    manifest = select_manifest(*metadata())
    monkeypatch.setattr(runner, "read_run", lambda root: ({}, manifest))
    calls = []
    def pilot(root, records, *a, **k):
        calls.append(records)
        return 1.0
    monkeypatch.setattr(runner, "cache_stage", pilot)
    with pytest.raises(RuntimeError, match="projected workload exceeds budget"):
        runner.run(tmp_path, ["0", "1", "2", "3"], "train-validation")
    assert len(calls) == 1 and len(calls[0]) == 4


@pytest.mark.slow
def test_physical_forecast_native_smoke():
    """Real T21/L12/MOLA solver smoke using native observations, never old forecasts."""
    pytest.importorskip("dinosaur")
    xr = pytest.importorskip("xarray")
    from cache_neural_temp import forecast
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "data/neural_macda").rglob("000000-000120.nc"))
    if not paths:
        pytest.skip("optional local native observations not staged")
    with xr.open_dataset(paths[0], decode_times=False) as ds:
        window = ds.isel(time=slice(0, 14)).load()
    forcing = physical_forcing()
    lead = forcing.rotation_period_s/12
    contract = {"lead_seconds": lead, "dt_seconds": lead/25, "steps": 25,
                "terrain_path": str(root / "data/mola/meg004/megt90n000cb.img")}
    record = {"start_sol": float(window.time[12]), "ls": float(window.Ls[12])}
    result = forecast(window, record, contract)
    assert result["features"].shape == (2048, len(SCHEMA))
    assert all(np.isfinite(v).all() for v in result.values())
    assert result["area"].sum() == pytest.approx(1)
    assert result["day"].any() and not result["day"].all()


def test_observed_initialization_and_future_predictor_isolation(monkeypatch):
    pytest.importorskip("dinosaur")
    xr = pytest.importorskip("xarray")
    from src.framework.gcm._dinosaur import jax
    jax.config.update("jax_enable_x64", True)
    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars import maps, topography
    from src.framework.physics.gcm import _true_anomaly
    from cache_neural_temp import forecast
    coords = coordinate_system("T21", 12)
    lon, lat = np.degrees(coords.horizontal.longitudes), np.degrees(coords.horizontal.latitudes)
    shape = (14, 12, len(lat), len(lon))
    spatial = shape[:1]+shape[2:]
    window = xr.Dataset({"temp": (("time", "lev", "lat", "lon"), np.full(shape, 215.)),
        "uwind": (("time", "lev", "lat", "lon"), np.zeros(shape)),
        "vwind": (("time", "lev", "lat", "lon"), np.zeros(shape)),
        **{n: (("time", "lat", "lon"), np.full(spatial, value)) for n, value in
           (("tsurf", 220.), ("psurf", 610.), ("coldust", .3), ("co2ice", 2.))}},
        coords={"time": np.arange(14)/12, "lev": coords.vertical.centers, "lat": lat, "lon": lon})
    forcing = physical_forcing()
    contract = {"lead_seconds": forcing.rotation_period_s/12, "dt_seconds": forcing.rotation_period_s/300,
                "steps": 25, "terrain_path": "unused"}
    seen = []
    def fake_run(**kwargs):
        seen.append(kwargs)
        return None, kwargs["initial_state"]
    monkeypatch.setattr(maps, "run_maps", fake_run)
    monkeypatch.setattr(topography, "regrid_to_nodal", lambda *a, **k: np.zeros((64, 32)))
    record = {"start_sol": 1., "ls": 80.}
    original = forecast(window, record, contract)
    altered = window.copy(deep=True)
    for name in altered.data_vars:
        altered[name].values[-1] += 100
    changed = forecast(altered, record, contract)
    for name in original.keys()-{"target"}:
        np.testing.assert_array_equal(original[name], changed[name])
    np.testing.assert_allclose(changed["target"]-original["target"], 100)
    np.testing.assert_allclose(original["features"][:, 7], 215., atol=1e-8)
    assert seen[0]["forcing"].pbl_diffusion_enabled
    assert seen[0]["forcing"].convective_adjustment_enabled
    np.testing.assert_allclose(seen[0]["forcing"].dust_visible_optical_depth, .3)
    f = seen[0]["forcing"]
    ls = float(_true_anomaly(1.5*f.rotation_period_s, f))+f.ls_perihelion_rad
    assert np.degrees(ls) % 360 == pytest.approx(80.)
    assert original["lst"][0] == pytest.approx(2.)


def test_native_loader_units_and_split(monkeypatch):
    pytest.importorskip("dinosaur")
    xr = pytest.importorskip("xarray")
    import stage_arco_macda
    from neural_temp_data import load_window, UNITS
    source = xr.Dataset({name: ("time", np.full(14, 1.), {"units": next(iter(units))}) for name, units in UNITS.items()},
                        coords={"time": ("time", np.arange(14)/12, {"units": "sol"}), "lev": ("lev", [.9], {"units": "1"})})
    source["Ls"] = ("time", np.ones(14), {"units": "degree"})
    source["MY_Ls"] = ("time", np.full(14, 24.), {"units": "1"})
    record = {"indices": list(range(14)), "times": source.time.values.tolist(), "year": 24}
    monkeypatch.setattr(stage_arco_macda, "_to_dinosaur_grid", lambda ds, *args: ds)
    assert len(load_window(source, record).time) == 14
    source.temp.attrs["units"] = "Celsius"
    with pytest.raises(ValueError, match="units"):
        load_window(source, record)
    source.temp.attrs["units"] = "K"
    source.MY_Ls.values[-1] = 25
    with pytest.raises(ValueError, match="split/year"):
        load_window(source, record)


def test_cli_fit_freeze_test_report_and_reuse(tmp_path):
    """Full 112/32/32 control workflow on synthetic cache, not a Mars skill test."""
    from neural_temp_data import implementation_hash
    from run_neural_temp import finalize
    manifest = select_manifest(*metadata())
    contract = {"manifest_hash": digest(manifest), "implementation_hash": implementation_hash(),
                "software": {}, "fixture": "synthetic optimizer/integration test, no scientific result"}
    write_json(tmp_path / "contract.json", contract)
    write_json(tmp_path / "manifest.json", manifest)
    data = synthetic(176, cells=16)
    for i, record in enumerate(manifest["starts"]):
        cache_entry(tmp_path, record, contract, {k: v[i] for k, v in data.items()})
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    def command(name, *extra):
        result = subprocess.run([sys.executable, str(scripts/name), "--run-dir", str(tmp_path), *extra],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
        return result.stdout
    command("train_neural_temp.py")
    frozen = verify_frozen(tmp_path, contract, require_gate=True)
    assert frozen["validation_gate"]
    assert "already frozen" in command("train_neural_temp.py")
    command("evaluate_neural_temp.py")
    seal = (tmp_path / "test_complete.json").read_bytes()
    assert "already evaluated" in command("evaluate_neural_temp.py")
    assert seal == (tmp_path / "test_complete.json").read_bytes()
    report = json.loads((tmp_path / "test_report.json").read_text())
    assert len(report["metrics"]) == 6
    assert len(report["ids"]) == 32
    write_json(tmp_path / "result.json", {"status": "synthetic_test_complete"})
    finalize(tmp_path, contract, manifest, tested=True)
    assert "synthetic_test_complete" in command("run_neural_temp.py", "--verify")
