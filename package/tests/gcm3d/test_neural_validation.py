"""Small holdout selection, scheduling and numerical validation guards."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_neural_pbl as training


def test_schedule():
    due = training.validation_due
    assert due(30, 10, 3, 0, 100)
    assert not due(20, 10, 3, 0, 100)
    assert due(25, 10, 3, 25, 100)
    assert due(100, 10, 0, 0, 100)
    assert not due(31, 10, 3, 25, 100)


def test_validation_is_bounded_and_held_out(monkeypatch):
    accessed = []
    produced = []

    class Cache:
        def iterate(self, selections, split):
            assert split == "validation"
            accessed.extend(selections)
            yield from selections

    def examples(experiment, ds, warmup):
        for i in range(100):
            produced.append(i)
            yield np.array([ds[0], i])

    monkeypatch.setattr(training, "examples", examples)
    subset, selections = training.validation_subset(Cache(), None, None, 4)
    assert len(subset) == len(produced) == 4
    assert accessed == selections and len(selections) == 2
    assert all(start >= 64185 and stop <= 80231 for start, stop in selections)
    assert produced == [0, 1, 0, 1]


def test_report_does_not_update_parameters_and_rejects_nonfinite():
    params = np.array([2.])
    report = training.validation_report(params, [1., 3.], lambda p, x: p[0] * x,
                                        [5., 7.], 20, 10)
    assert report["neural_loss"] == 4. and report["physical_loss"] == 6.
    assert report["epoch"] == 2.
    np.testing.assert_array_equal(params, [2.])
    with pytest.raises(ValueError, match="nonfinite"):
        training.validation_report(params, [1.], lambda p, x: np.nan, [1.], 0, 10)
