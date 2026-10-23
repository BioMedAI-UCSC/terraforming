"""A naturally evolving forecast must not be confused with control improvement."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dinosaur")


def test_zero_control_uses_final_forecast_not_initial_state(monkeypatch):
    examples = Path(__file__).resolve().parents[3] / "examples/neural"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("control_example", examples / "optimize_atmospheric_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = np.array([210., 200.])
    forecast = np.array([215., 202.])
    target = forecast - 1.5
    metrics = module.control_metrics(initial, forecast, forecast, target)
    assert metrics["initial_rmse_k"] == pytest.approx(1.5)
    assert metrics["uncontrolled_final_rmse_k"] == pytest.approx(1.5)
    assert metrics["rmse_improvement_k"] == pytest.approx(0.)
    assert metrics["initial_state_rmse_k"] != pytest.approx(1.5)
    controlled = module.control_metrics(initial, forecast, target, target)
    assert controlled["rmse_improvement_k"] == pytest.approx(1.5)
