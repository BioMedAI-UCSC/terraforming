from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from diagnose_neural_pbl import correction_summary, error_metrics, shifted, constant_variant, constant_screen


def test_layer_metrics_match_training_weighting():
    predicted = np.arange(2 * 3 * 4 * 2 * 2).reshape(2, 3, 4, 2, 2) / 10.
    target = np.ones_like(predicted)
    mass = np.arange(2 * 4 * 2 * 2).reshape(2, 4, 2, 2) + 1.
    result = error_metrics(predicted, target, mass)
    expected = np.sum(((predicted - target) / 10)**2 * mass[:, None]) / (3 * mass.sum())
    assert result["loss"] == pytest.approx(expected)
    assert result["layer_mse"].shape == (3, 4)
    lower = np.sum(((predicted[:, :, -3:] - target[:, :, -3:]) / 10)**2 * mass[:, None, -3:]) / (3 * mass[:, -3:].sum())
    assert result["lowest_three_layer_loss"] == pytest.approx(lower)
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


def test_constant_screen_preserves_architecture_and_selects_baselines():
    from src.framework.physics.neural_pbl import initialize, multipliers, Closure
    from src.framework.gcm._dinosaur import jax, jnp
    for constant in (False, True):
        params = initialize(jax.random.key(0), constant=constant)
        changed = constant_variant(params, [-8., 8.])
        assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(changed)
        features = jax.random.normal(jax.random.key(1), (4, 13))
        factors = multipliers(Closure(changed, jnp.zeros(13), jnp.ones(13)), features)
        np.testing.assert_allclose(factors, np.tile([.5, 2.], (4, 1)), atol=1e-6)
        np.testing.assert_array_equal(params["constant" if constant else "b3"], 0.)
    aggregate = {"physical": {"loss": 10., "lowest_three_layer_loss": 5.},
                 "neural": {"loss": 8., "lowest_three_layer_loss": 3.},
                 "constant_a": {"loss": 9., "lowest_three_layer_loss": 6.},
                 "constant_b": {"loss": 11., "lowest_three_layer_loss": 4.}}
    screen = constant_screen(aggregate)
    assert screen["best_global"] == "constant_a"
    assert screen["best_lowest_three_layers"] == "constant_b"
    assert screen["global_improvement_percent"] == pytest.approx(10.)
