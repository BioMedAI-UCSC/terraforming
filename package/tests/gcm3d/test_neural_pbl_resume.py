"""PBL cursor recovery across excluded source chunks using the real trainer loop."""
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_neural_pbl as trainer
from src.framework.gcm._dinosaur import jax, jnp


def test_pbl_resume_preserves_cursor_across_excluded_windows(tmp_path, monkeypatch):
    selections = [(0, 5), (5, 10), (10, 15)]
    class Cache:
        def iterate(self, selections, split, *, indexed=False):
            assert indexed
            for window in selections:
                yield window, None if window[0] == 5 else window

    class Experiment:
        dt = 1.
        def __init__(self, *args):
            pass
        def loss(self, params, batch, *args):
            return jnp.mean((params["constant"] - batch)**2)

    monkeypatch.setattr(trainer, "Experiment", Experiment)
    monkeypatch.setattr(trainer, "windows", lambda split: selections)
    monkeypatch.setattr(trainer, "parallel_prepare", lambda *args: lambda batch: batch)
    monkeypatch.setattr(trainer, "validation_subset", lambda *args: ([jnp.ones(2)], [(20, 25)]))
    def examples(experiment, window, warmup, skip=0, **kwargs):
        for i in range(skip, 4):
            yield jnp.full(2, (window[0] + i + 1.) / 10.)
    monkeypatch.setattr(trainer, "examples", examples)

    def args(output, steps):
        return SimpleNamespace(output=output, steps=steps, revision="synthetic", seed=5,
            constant=True, max_dt=1., horizon=1, spinup=0, normalization_chunks=1,
            learning_rate=.01, regularization=0., devices=1, chunks=None, resume=True,
            evaluate=None, validation_examples=1, preflight_only=False,
            validate_every_epochs=1, validate_every_steps=0)

    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    for folder in (continuous, resumed):
        folder.mkdir()
        options = args(folder, 12)
        contract = {key: getattr(options, key) for key in (
            "revision", "seed", "constant", "max_dt", "horizon", "spinup",
            "normalization_chunks", "learning_rate", "regularization", "devices", "chunks")}
        contract.update(features=list(trainer.FEATURE_NAMES), version=1)
        zeros = {"constant": np.zeros(2)}
        saved = dict(contract=contract, params=zeros, first=zeros, second=zeros,
                     mean=np.zeros(13), scale=np.ones(13), completed=0, position=0,
                     order=[0, 1, 2], rng=np.random.default_rng(5).bit_generator.state)
        (folder / "checkpoint.pkl").write_bytes(pickle.dumps(saved))
    devices = jax.local_devices()[:1]
    trainer.run(args(continuous, 12), Cache(), devices)
    # Stop just before an exclusion, just after it, then at the epoch boundary.
    for steps in (4, 5, 8, 12):
        trainer.run(args(resumed, steps), Cache(), devices)
    a, b = [pickle.loads((folder / "checkpoint.pkl").read_bytes()) for folder in (continuous, resumed)]
    for key in ("completed", "position", "order", "rng", "contract"):
        assert a[key] == b[key]
    for key in ("params", "first", "second", "mean", "scale"):
        for x, y in zip(jax.tree_util.tree_leaves(a[key]), jax.tree_util.tree_leaves(b[key])):
            np.testing.assert_array_equal(x, y)
    logs = [[json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
            for folder in (continuous, resumed)]
    for x, y in zip(*logs, strict=True):
        for key in ("step", "position", "loss", "gradient_norm"):
            assert x[key] == y[key]
