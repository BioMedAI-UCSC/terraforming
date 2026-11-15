"""Atomic, pickle-free reference-MLP checkpoints with explicit compatibility."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from src.framework.gcm._dinosaur import jnp
from .models import Normalization


def _validate(params, normalization, model):
    model.apply(params, jnp.zeros((1, model.input_size)))
    for value in (*[a for pair in params for a in pair], *normalization):
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("checkpoint contains nonfinite arrays")
    if normalization.mean.shape != (model.input_size,) or normalization.scale.shape != (model.input_size,):
        raise ValueError("normalization shape mismatch")
    if np.any(np.asarray(normalization.scale) <= 0):
        raise ValueError("normalization scales must be positive")


def save_checkpoint(path, params, normalization, model, *, metadata):
    """Save weights and statistics; metadata must identify features and physics.

    This is a frozen inference checkpoint, not an optimizer-resume checkpoint.
    User-defined model architectures own their serialization.
    """
    if not metadata.get("feature_schema") or not metadata.get("physics"):
        raise ValueError("checkpoint requires feature_schema and physics metadata")
    _validate(params, normalization, model)
    info = {"version": 1, "input_size": model.input_size, "output_size": model.output_size,
            "hidden_sizes": list(model.hidden_sizes), "metadata": metadata}
    arrays = {f"{name}_{i}": np.asarray(value) for i, pair in enumerate(params)
              for name, value in zip(("weight", "bias"), pair)}
    arrays.update(mean=np.asarray(normalization.mean), scale=np.asarray(normalization.scale),
                  metadata=np.asarray(json.dumps(info, sort_keys=True, allow_nan=False)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, *, expected_metadata):
    """Reject architecture, physical configuration and feature-schema mismatches."""
    expected = {"version": 1, "input_size": model.input_size, "output_size": model.output_size,
                "hidden_sizes": list(model.hidden_sizes), "metadata": expected_metadata}
    expected = json.loads(json.dumps(expected, allow_nan=False))
    def restore(array):
        value = jnp.asarray(array)
        if value.dtype != array.dtype:
            raise ValueError("checkpoint dtype changed on load; enable JAX float64 for float64 checkpoints")
        return value

    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"])) != expected:
            raise ValueError("checkpoint configuration mismatch")
        params = tuple((restore(archive[f"weight_{i}"]), restore(archive[f"bias_{i}"]))
                       for i in range(len(model.hidden_sizes) + 1))
        norm = Normalization(restore(archive["mean"]), restore(archive["scale"]))
    _validate(params, norm, model)
    return params, norm
