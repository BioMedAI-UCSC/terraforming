"""Exercise the real trainer/checkpoint loop with inexpensive deterministic physics."""
import dataclasses
from collections import namedtuple
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_neural_hybrid as trainer
from src.framework.gcm._dinosaur import jax, jnp

Dynamics = namedtuple("Dynamics", "log_surface_pressure sim_time", defaults=(0.,))
State = namedtuple("State", "dynamics")


@dataclasses.dataclass
class Chunk:
    start: int
    sizes: dict


class Cache:
    def iterate(self, selections, split, *, indexed=False):
        for start, stop in selections:
            ds = None if start == 10 else Chunk(start, {"time": stop - start})
            if indexed:
                yield (start, stop), ds
            elif ds is not None:
                yield ds


class ToyExperiment:
    def __init__(self, *args, **kwargs):
        self.terrain = np.zeros((1, 1))
        self.dt, self.refresh_steps, self.pressure_scale = 1., 1, 1.
        self.coords = SimpleNamespace(
            vertical=SimpleNamespace(layers=1, boundaries=np.array([0., 1.])),
            horizontal=SimpleNamespace(quadrature_weights=np.ones((1, 1)), to_nodal=lambda x: x))

    def loss(self, params, example, stats, horizon, regularization):
        return jnp.mean((params["physics"]["decode"]["b"][None, :, None, None, None] - example[2][:horizon])**2)

    def metrics(self, params, example, stats, horizon, physical=False, return_forecasts=False):
        target = example[2][:horizon]
        bias = jnp.zeros(3) if physical else params["physics"]["decode"]["b"]
        prediction = jnp.broadcast_to(bias[None, :, None, None, None], target.shape)
        mse = jnp.mean((prediction - target)**2)
        metrics = mse, mse, mse, jnp.ones((3, 1)) * jnp.sqrt(mse)
        return (*metrics, prediction, prediction) if return_forecasts else metrics

    def output(self, *args, **kwargs):
        return jnp.zeros((3, 1, 1, 1))


@pytest.mark.parametrize("export_predictions", [False, True])
def test_trainer_resume_across_curriculum_and_epoch(tmp_path, monkeypatch, export_predictions):
    monkeypatch.setattr(trainer, "HybridExperiment", ToyExperiment)
    monkeypatch.setattr(trainer, "windows", lambda split:
                        [(0, 5), (5, 10), (10, 15)] if split == "train" else [(0, 5), (5, 10)])
    normalizations = []

    def statistics(e, chunks):
        normalizations.append(list(chunks))
        return {"mean": np.zeros(1), "scale": np.ones(1), "field_scale": np.ones((3, 1))}

    monkeypatch.setattr(trainer, "fit_statistics", statistics)

    def examples(e, ds, horizon, skip=0):
        for i in range(skip, ds.sizes["time"] - horizon):
            yield (State(Dynamics(jnp.zeros((1, 1, 1)), jnp.asarray(float(ds.start + i)))), jnp.zeros(1),
                   jnp.full((horizon, 3, 1, 1, 1), (ds.start + i + 1.) / 10.), jnp.ones((horizon, 1, 1, 1)))

    monkeypatch.setattr(trainer, "examples", examples)

    def arguments(output, steps, **overrides):
        output.mkdir(exist_ok=True)
        return SimpleNamespace(**(dict(
            output=output, steps=steps, revision="synthetic", seed=5, width=2, blocks=1,
            max_dt=1., refresh_seconds=1., spinup=0, mola=None, flat_terrain=True,
            learning_rate=.01, regularization=0., normalization_chunks=2, chunks=None,
            devices=1, curriculum=[(1, 2), (2, 8)], ablation="full",
            validation_horizon=1, validation_examples=2, resume=False, evaluate=None,
            preflight_only=False, validate_every_steps=2, validate_every_epochs=1,
            export_validation_predictions=export_predictions) | overrides))

    devices = jax.local_devices()[:1]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    trainer.run(arguments(continuous, 8), Cache(), devices)
    trainer.run(arguments(resumed, 8, preflight_only=True), Cache(), devices)
    assert pickle.loads((resumed / "checkpoint.pkl").read_bytes())["completed"] == 0
    trainer.run(arguments(resumed, 3, resume=True), Cache(), devices)
    trainer.run(arguments(resumed, 6, resume=True), Cache(), devices)
    trainer.run(arguments(resumed, 8, resume=True), Cache(), devices)
    assert len(normalizations) == 2  # Resume never refits frozen statistics.
    a, b = [pickle.loads((path / "checkpoint.pkl").read_bytes()) for path in (continuous, resumed)]
    for key in ("contract", "completed", "position", "order", "rng"):
        assert a[key] == b[key]
    for key in ("params", "first", "second", "stats"):
        for x, y in zip(jax.tree_util.tree_leaves(a[key]), jax.tree_util.tree_leaves(b[key])):
            np.testing.assert_array_equal(x, y)
    logs = [[json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]
            for path in (continuous, resumed)]
    for x, y in zip(*logs, strict=True):
        for key in ("step", "horizon", "loss", "gradient_norm"):
            assert x[key] == y[key]
    if export_predictions:
        from neural_screen_metrics import ensemble_report
        artifact = resumed / "validation-predictions/step-000008.npz"
        report = ensemble_report([artifact] * 4)
        validation = json.loads((resumed / "small_validation.json").read_text())
        assert report["neural_loss"] == pytest.approx(validation["neural_loss"])
        assert report["physical_loss"] == pytest.approx(validation["physical_loss"])
    with pytest.raises(ValueError, match="contract differs"):
        trainer.run(arguments(resumed, 8, resume=True, ablation="adapters-only"), Cache(), devices)

    # A finite decoded score must not hide a broken internal forecast.
    checkpoint_before = (resumed / "checkpoint.pkl").read_bytes()
    monkeypatch.setattr(ToyExperiment, "metrics", lambda *args, **kwargs:
                        (jnp.asarray(1.), jnp.asarray(0.), jnp.asarray(float("nan")), jnp.ones((3, 1)),
                         jnp.zeros((1, 3, 1, 1, 1)), jnp.zeros((1, 3, 1, 1, 1))))
    with pytest.raises(ValueError, match="nonfinite validation"):
        trainer.run(arguments(resumed, 8, resume=True), Cache(), devices)
    assert (resumed / "checkpoint.pkl").read_bytes() == checkpoint_before
