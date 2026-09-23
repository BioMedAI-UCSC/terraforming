"""GPU routing, worker supervision, and paired forecast ensemble regression tests."""
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from neural_screen_metrics import ensemble_report, forecast_scores
from run_neural_screen import METHODS, Table, build_jobs, comparison_contract, parse_args, run_jobs


def artifact(path, prediction, *, step=5, metadata="{}", target=None, mass=None):
    target = np.zeros_like(prediction) if target is None else target
    mass = np.ones(prediction.shape[:2] + prediction.shape[3:]) if mass is None else mass
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction, target=target, mass=mass,
                        field_scale=np.ones(prediction.shape[2:4]),
                        physical_losses=np.full(prediction.shape[0], 4.), latent_losses=np.full(prediction.shape[0], 2.),
                        step=step, metadata=metadata)


def test_ensemble_averages_forecasts_not_losses(tmp_path):
    paths = [tmp_path / f"{i}.npz" for i in range(4)]
    prediction = np.ones((2, 1, 3, 2, 2, 2))
    for path, factor in zip(paths, (-2, -1, 1, 2)):
        artifact(path, prediction * factor)
    report = ensemble_report(paths)
    assert report["neural_loss"] == 0.
    assert report["member_losses"] == [4., 1., 1., 4.]
    assert report["improvement_percent"] == 100.
    assert report["weights"] == [.25] * 4
    assert report["case_losses"] == [0., 0.]
    prediction[0] *= 2
    mass = np.arange(2 * 1 * 2 * 2 * 2).reshape(2, 1, 2, 2, 2) + 1.
    result = forecast_scores(prediction, np.zeros_like(prediction), mass, np.ones((3, 2)))
    assert result["case_losses"] == [4., 1.]
    assert result["neural_loss"] == 2.5  # Equal case weighting, not pooled mass weighting.


@pytest.mark.parametrize("key", ["step", "metadata", "target", "mass"])
def test_ensemble_refuses_unpaired_cases(tmp_path, key):
    paths = [tmp_path / f"{i}.npz" for i in range(4)]
    prediction = np.ones((1, 1, 3, 2, 2, 2))
    for path in paths:
        artifact(path, prediction)
    change = dict(step=6, metadata='{"seed": 1}', target=prediction,
                  mass=np.full((1, 1, 2, 2, 2), 2.))[key]
    artifact(paths[-1], prediction, **{key: change})
    with pytest.raises(ValueError, match=key):
        ensemble_report(paths)


def test_commands_route_one_gpu_each_and_keep_comparison_fixed(tmp_path):
    args = parse_args(["--flat-terrain", "--output", str(tmp_path), "--gpus", "3,1,0,2", "--dry-run"])
    jobs = build_jobs(args)
    assert [job["method"] for job in jobs] == list(METHODS)
    assert [job["env"]["CUDA_VISIBLE_DEVICES"] for job in jobs] == ["3", "1", "0", "2"]
    commands = []
    for job in jobs:
        command = job["command"]
        assert command[command.index("--devices") + 1] == "1"
        assert command[command.index("--ablation") + 1] == job["method"]
        assert job["env"]["JAX_PLATFORMS"] == "cuda"
        assert "--export-validation-predictions" in command
        comparable = command.copy()
        comparable[comparable.index("--output") + 1] = "OUTPUT"
        comparable[comparable.index("--ablation") + 1] = "METHOD"
        commands.append(comparable)
    assert all(command == commands[0] for command in commands)
    contract = comparison_contract(args)
    args.steps, args.resume = 100, True
    assert comparison_contract(args) == contract
    assert all("--resume" in job["command"] for job in build_jobs(args))
    args.width = 64
    assert comparison_contract(args) != contract
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "0,0,1,2", "--dry-run"])


def fake_jobs(tmp_path, *, fail=None, sleep=False):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import json, os, pathlib, signal, sys, time
method, gpu, output, fail, sleep = sys.argv[1:]
assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
if sleep == "True":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    time.sleep(30)
if method == fail:
    print("test worker failed", flush=True)
    sys.exit(7)
import numpy as np
path = pathlib.Path(output) / "validation-predictions" / "step-000005.npz"
path.parent.mkdir(parents=True)
prediction = np.ones((1, 1, 3, 2, 2, 2))
np.savez_compressed(path, prediction=prediction, target=np.zeros_like(prediction),
                    mass=np.ones((1, 1, 2, 2, 2)), field_scale=np.ones((3, 2)),
                    physical_losses=np.array([4.]), latent_losses=np.array([2.]), step=5, metadata="{}")
print(json.dumps(dict(event="training_progress", step=5, loss=1., gradient_norm=.1)), flush=True)
print(json.dumps(dict(event="small_validation", step=5, neural_loss=1., physical_loss=4.)), flush=True)
''')
    return [dict(method=method, gpu=str(i), folder=tmp_path / method,
                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
                 command=[sys.executable, str(worker), method, str(i), str(tmp_path / method), str(fail), str(sleep)])
            for i, method in enumerate(METHODS)]


@pytest.mark.parametrize("fail", [None, "decoder-only"])
def test_concurrent_workers_and_failure_reporting(tmp_path, fail):
    stream = io.StringIO()
    report = run_jobs(fake_jobs(tmp_path, fail=fail), tmp_path, 1., Table([stream]), grace_seconds=.2)
    assert (tmp_path / "summary.json").exists()
    assert all((tmp_path / method / "console.log").exists() for method in METHODS)
    assert "| GPU |" in stream.getvalue()
    if fail:
        assert report["methods"][fail]["returncode"] == 7
        assert report["methods"][fail]["status"] == "failed"
        assert report["ensembles"] == {}
        assert sum(item["status"] == "complete" for item in report["methods"].values()) == 3
    else:
        assert all(item["status"] == "complete" for item in report["methods"].values())
        assert report["ensembles"]["5"]["neural_loss"] == 1.
        assert "Paired validation at step 5" in stream.getvalue()


def test_wall_time_stops_workers_and_saves_status(tmp_path):
    started = time.monotonic()
    report = run_jobs(fake_jobs(tmp_path, sleep=True), tmp_path, .005, Table([io.StringIO()]), grace_seconds=.2)
    assert time.monotonic() - started < 10
    assert report["timed_out"]
    assert all(item["status"] == "stopped" for item in report["methods"].values())
    assert all(item["returncode"] is not None for item in report["methods"].values())
