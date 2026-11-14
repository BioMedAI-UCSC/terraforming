from pathlib import Path
import sys

import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from run_differentiable_experiments import observation_means, target_arrays, d
from prepare_multiday_windows import daily_sample_times


def test_daily_means_match_sampling_and_keep_derivatives():
    values = np.arange(24., dtype=float).reshape(24, 1)
    np.testing.assert_array_equal(observation_means(values, {}), values)
    row = dict(samples_per_observation=12)
    np.testing.assert_allclose(observation_means(values, row).ravel(), [5.5, 17.5])
    gradient = d.jax.jacfwd(lambda scale: observation_means(d.jnp.asarray(values) * scale, row).sum())(1.)
    assert float(gradient) == pytest.approx(23.)


def test_five_sol_daily_sampling_has_no_initialization_overlap():
    centres, samples, dt = daily_sample_times(88775.244)
    assert len(centres) == 5 and len(samples) == 60
    assert samples[0] / 88775.244 == pytest.approx(1. / 12.)
    assert samples[-1] / 88775.244 == pytest.approx(5., abs=1e-12)
    np.testing.assert_allclose(observation_means(samples, dict(samples_per_observation=12)), centres)
    np.testing.assert_allclose(samples / dt, np.rint(samples / dt), atol=1e-9)


def test_daily_targets_validate_centres_and_array_grouping(tmp_path, monkeypatch):
    centres, samples, dt = daily_sample_times(88775.244)
    target = tmp_path / "target.npz"
    fields = np.zeros((5, 3, 12, 64, 32))
    np.savez(target, seconds=centres, sample_seconds=samples, fields=fields)
    row = dict(target=str(target), restart="unused", start_seconds=0.,
               end_seconds=5.5 * 88775.244, samples_per_observation=12)
    model = SimpleNamespace(base_forcing=None, diagnostics=lambda state: None)
    monkeypatch.setattr(d, "load_restart", lambda path: "initial")
    selected, actual = target_arrays(row, model, dt)
    assert len(selected) == 60 and actual.shape == fields.shape
    np.savez(target, seconds=centres + 1., sample_seconds=samples, fields=fields)
    with pytest.raises(ValueError, match="centres"):
        target_arrays(row, model, dt)
    with pytest.raises(ValueError, match="sample count"):
        target_arrays({**row, "samples_per_observation": 1}, model, dt)
