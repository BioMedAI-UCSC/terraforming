"""Accuracy budgets must not silently depend on timer or machine speed."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dinosaur")


def test_fixed_updates_ignore_wall_clock_and_repeat(monkeypatch):
    folder = Path(__file__).resolve().parents[3] / "examples/neural"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("radiation_example", folder / "train_radiation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = lambda p: module.jnp.sum((p - 2.)**2)
    initial = module.jnp.asarray([0.])
    first, a = module.fit(initial, loss, loss, 7, .01)
    # A vastly different clock must not change the update count or checkpoint.
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1e9)
    second, b = module.fit(initial, loss, loss, 7, .01)
    np.testing.assert_array_equal(first, second)
    assert a["updates"] == b["updates"] == 7
    assert a["budget_mode"] == b["budget_mode"] == "fixed_updates"
    assert a["selected_update"] == b["selected_update"] == 7


def test_validation_can_select_initial_checkpoint(monkeypatch):
    folder = Path(__file__).resolve().parents[3] / "examples/neural"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("radiation_example", folder / "train_radiation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = module.jnp.asarray([0.])
    best, report = module.fit(initial, lambda p: module.jnp.sum((p-2.)**2),
                              lambda p: module.jnp.sum(p**2), 3, .01)
    np.testing.assert_array_equal(best, initial)
    assert report["selected_update"] == 0
