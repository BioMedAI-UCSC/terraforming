from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from diagnose_neural_pbl import correction_summary, error_metrics, shifted


def test_layer_metrics_match_training_weighting():
    predicted = np.arange(2 * 3 * 4 * 2 * 2).reshape(2, 3, 4, 2, 2) / 10.
    target = np.ones_like(predicted)
    mass = np.arange(2 * 4 * 2 * 2).reshape(2, 4, 2, 2) + 1.
    result = error_metrics(predicted, target, mass)
    expected = np.sum(((predicted - target) / 10)**2 * mass[:, None]) / (3 * mass.sum())
    assert result["loss"] == pytest.approx(expected)
    assert result["layer_mse"].shape == (3, 4)
    assert error_metrics(target, target, mass)["loss"] == 0.


def test_saturation_and_perturbations():
    result = correction_summary([.5, 1., 2.])
    assert result["fraction_near_lower_bound"] == pytest.approx(1 / 3)
    assert result["fraction_near_upper_bound"] == pytest.approx(1 / 3)
    for key in ("b3", "constant"):
        params = {key: np.zeros(2)}
        changed = shifted(params, 1, .1)
        np.testing.assert_array_equal(params[key], [0., 0.])
        np.testing.assert_array_equal(changed[key], [0., .1])
