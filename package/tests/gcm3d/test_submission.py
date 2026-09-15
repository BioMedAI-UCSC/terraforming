"""Scientific metadata and metric regressions for submission tooling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gaussian_area_mean_integrates_latitude_polynomial():
    benchmark = _script("benchmark_mcd")
    nodes, _ = np.polynomial.legendre.leggauss(32)
    lat = np.degrees(np.arcsin(nodes))
    model = np.broadcast_to(nodes[:, None] ** 2, (32, 64))
    result = benchmark.weighted_metrics(model, np.zeros_like(model), lat)
    assert result["model_area_mean"] == pytest.approx(1 / 3, abs=1e-14)
    with pytest.raises(ValueError, match="No valid"):
        benchmark.weighted_metrics(model, np.full_like(model, np.nan), lat)


def test_checkpoint_history_rejects_reversed_time(tmp_path):
    driver = _script("run_gcm3d_ablation")
    path = tmp_path / "history.csv"
    driver._append_checkpoint_diagnostics(path, {"step": 10, "value": 1.0})
    driver._append_checkpoint_diagnostics(path, {"step": 10, "value": 1.0})
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="ahead of restart"):
        driver._append_checkpoint_diagnostics(path, {"step": 9, "value": 1.0})
