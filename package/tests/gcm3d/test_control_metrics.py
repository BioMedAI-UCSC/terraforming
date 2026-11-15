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


def test_control_keeps_best_iterate_including_initial_and_checks_final(monkeypatch):
    examples = Path(__file__).resolve().parents[3] / "examples/neural"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("control_fit", examples / "optimize_atmospheric_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = lambda p: module.jnp.sum(p**2)
    # Adam's fixed step overshoots this small target; the last step is worse.
    best, report = module.fit_control(module.jnp.asarray([0.01]), loss, 1)
    assert report["final_objective"] <= report["initial_objective"]
    assert report["final_objective"] <= report["last_iterate_objective"]
    assert float(loss(best)) == pytest.approx(report["final_objective"])
    initial = module.jnp.zeros(1)
    best, report = module.fit_control(initial, loss, 3)
    np.testing.assert_array_equal(best, initial)
    assert report["selected_update"] == 0
    best, report = module.fit_control(module.jnp.asarray([1.]), loss, 1)
    assert report["selected_update"] == 1
    assert report["final_objective"] < report["initial_objective"]
    with pytest.raises(FloatingPointError, match="initial"):
        module.fit_control(initial, lambda p: module.jnp.sum(p) * module.jnp.nan, 1)
    monkeypatch.setattr(module, "adam_update", lambda p, grad, opt, **kwargs: (p + 1., opt))
    best, report = module.fit_control(initial, lambda p: module.jnp.sum((p - 1.)**2), 3)
    np.testing.assert_array_equal(best, [1.])
    assert report["selected_update"] == 1
    assert report["final_objective"] == 0.
    assert report["last_iterate_objective"] == 4.
