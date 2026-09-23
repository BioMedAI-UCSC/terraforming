"""Small JAX-only reference model and training-only normalization."""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp


class Normalization(NamedTuple):
    mean: object
    scale: object

    def apply(self, x):
        if x.shape[-1] != self.mean.shape[-1]:
            raise ValueError("feature count does not match normalization")
        return (x - self.mean) / self.scale


def fit_normalization(x, *, split, weights=None):
    """Fit on training rows only; optional weights must be finite and nonnegative."""
    if split != "train":
        raise ValueError("normalization may only be fitted on the train split")
    x = np.asarray(x)
    if x.ndim < 2 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite feature rows")
    flat = x.reshape(-1, x.shape[-1])
    w = np.ones(flat.shape[0]) if weights is None else np.broadcast_to(weights, x.shape[:-1]).reshape(-1)
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid normalization weights")
    w = w / w.sum()
    mean = np.sum(w[:, None] * flat, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (flat - mean)**2, axis=0))
    return Normalization(jnp.asarray(mean), jnp.asarray(np.maximum(scale, 1e-6)))


@dataclass(frozen=True)
class ColumnMLP:
    """Feature-last GELU MLP. Parameters are tuples of (weight, bias) arrays."""
    input_size: int
    output_size: int
    hidden_sizes: tuple[int, ...] = (32, 32)

    def __post_init__(self):
        if any(n < 1 for n in (self.input_size, *self.hidden_sizes, self.output_size)):
            raise ValueError("all model dimensions must be positive")

    def init(self, key):
        widths = (self.input_size, *self.hidden_sizes, self.output_size)
        keys = jax.random.split(key, len(widths) - 1)
        return tuple((jax.random.normal(k, (a, b)) * jnp.sqrt(1.0 / a), jnp.zeros(b))
                     for k, a, b in zip(keys, widths[:-1], widths[1:]))

    def apply(self, params, x):
        if x.shape[-1] != self.input_size:
            raise ValueError("model input feature count mismatch")
        widths = (self.input_size, *self.hidden_sizes, self.output_size)
        if len(params) != len(widths) - 1:
            raise ValueError("model layer count mismatch")
        for i, ((w, b), a, out) in enumerate(zip(params, widths[:-1], widths[1:])):
            if w.shape != (a, out) or b.shape != (out,):
                raise ValueError("model parameter shape mismatch")
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.gelu(x)
        return x
