"""Small optimizer and diagnostics; dataset splits and stopping policies stay outside."""
from typing import NamedTuple

from src.framework.gcm._dinosaur import jax, jnp


def weighted_mse(predicted, reference, weights=None):
    """Mean square error; weights broadcast over all prediction axes explicitly."""
    error = (predicted - reference)**2
    if weights is None:
        return jnp.mean(error)
    weights = jnp.broadcast_to(weights, error.shape)
    total = jnp.sum(weights)
    valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & (total > 0)
    return jnp.where(valid, jnp.sum(weights * error) / total, jnp.nan)


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(tree)))


class AdamState(NamedTuple):
    count: object
    first: object
    second: object


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def adam_update(params, grads, state, *, learning_rate=1e-3, clip_norm=1.0):
    """One Adam step with global norm clipping; NaNs remain visible to callers."""
    norm = tree_norm(grads)
    factor = jnp.minimum(1.0, clip_norm / jnp.maximum(norm, 1e-12))
    grads = jax.tree.map(lambda x: x * factor, grads)
    count = state.count + 1
    first = jax.tree.map(lambda m, g: .9*m + .1*g, state.first, grads)
    second = jax.tree.map(lambda v, g: .999*v + .001*g*g, state.second, grads)
    updated = jax.tree.map(
        lambda p, m, v: p - learning_rate * (m / (1-.9**count)) / (jnp.sqrt(v / (1-.999**count)) + 1e-8),
        params, first, second)
    return updated, AdamState(count, first, second)


def directional_gradient_check(loss, params, direction, epsilon=1e-4):
    """Host-side check of a scalar loss in a supplied parameter-space direction."""
    grads = jax.grad(loss)(params)
    autodiff = sum(jnp.vdot(g, d).real for g, d in zip(jax.tree.leaves(grads), jax.tree.leaves(direction)))
    plus = jax.tree.map(lambda p, d: p + epsilon*d, params, direction)
    minus = jax.tree.map(lambda p, d: p - epsilon*d, params, direction)
    finite_difference = (loss(plus) - loss(minus)) / (2*epsilon)
    return {"autodiff": float(autodiff), "finite_difference": float(finite_difference),
            "relative_error": float(jnp.abs(autodiff-finite_difference) /
                                    jnp.maximum(jnp.maximum(jnp.abs(autodiff), jnp.abs(finite_difference)), 1e-12))}
