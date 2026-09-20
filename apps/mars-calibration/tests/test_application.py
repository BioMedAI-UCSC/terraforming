import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mars_calibration import driver, launch


def test_exact_budgets_and_finite_differences(tmp_path, monkeypatch):
    # A flat oracle forces early optimizer termination, exercising budget restarts.
    # No coupled rollout is compiled in this application control-flow test.
    source = tmp_path / "input"
    source.write_text("fixture")
    target = {name: np.zeros((12, 64, 32)) for name in
              ("temperature", "eastward_wind", "northward_wind")}
    target.update(ls_deg=45.0, mars_year=32.0, time_sol=5443.0)
    monkeypatch.setattr(driver, "load_restart", lambda _: np.array([1.0]))
    monkeypatch.setattr(driver, "_target", lambda *_: target)
    monkeypatch.setattr(driver, "_objective_factory", lambda *args: lambda raw: driver.jnp.sum(raw * 0) + 1)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["calibrate", *([str(source)] * 4), "--output", str(report)])
    assert driver.main() == 10  # Completed, but never claim scientific success.
    result = json.loads(report.read_text())
    assert result["steps"] == 74
    assert result["lbfgs"]["evaluations"] == result["powell"]["evaluations"] == 20
    assert result["budget"]["exact"]
    assert result["budget"]["lbfgs_restarts"] > 0
    assert result["budget"]["finite_difference_forward_calls"] == 12
    assert all(check["pass"] for check in result["gradient_checks"])
    assert len(report.with_suffix(".jsonl").read_text().splitlines()) == 40


def test_bundle_integrity_and_canonical_settings(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / "experiment.json").read_text())
    manifest = {"files": {}}
    for name in launch.FILES:
        (tmp_path / name).write_bytes(b"test")
        manifest["files"][name] = {"sha256": launch.sha256(tmp_path / name)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert launch.validate(config, tmp_path) == manifest
    with pytest.raises(ValueError, match="max_evaluations"):
        launch.validate({**config, "max_evaluations": 19}, tmp_path)
    (tmp_path / "mola.img").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        launch.validate(config, tmp_path)
