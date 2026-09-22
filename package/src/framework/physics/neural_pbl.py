"""Bounded, identity-initialized interface closure; all inputs are SI quantities.

The architecture is 13 -> 32 -> 32 -> 2. Feature definitions here are explicit
project choices, not a claim to reproduce an unspecified paper's features.
"""
from typing import NamedTuple

from src.framework.gcm._dinosaur import jax, jnp

FEATURE_NAMES = (
    "temperature_upper", "temperature_lower", "u_upper", "u_lower",
    "v_upper", "v_lower", "theta_upper", "theta_lower", "friction_velocity",
    "richardson", "interface_height", "layer_separation", "sigma_upper",
)


class Closure(NamedTuple):
    params: object
    mean: object
    scale: object


def interface_features(*values):
    return jnp.stack(jnp.broadcast_arrays(*values), axis=-1)


def initialize(key, *, constant=False):
    """Zero output layer guarantees exactly unit multipliers for every input."""
    if constant:
        return {"constant": jnp.zeros(2)}
    keys = jax.random.split(key, 2)
    return {
        "w1": jax.random.normal(keys[0], (13, 32)) / jnp.sqrt(13.),
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(keys[1], (32, 32)) / jnp.sqrt(32.),
        "b2": jnp.zeros(32), "w3": jnp.zeros((32, 2)), "b3": jnp.zeros(2),
    }


def multipliers(closure, features):
    p = closure.params
    if "constant" in p:
        logits = jnp.broadcast_to(p["constant"], features.shape[:-1] + (2,))
    else:
        x = (features - closure.mean) / closure.scale
        x = jnp.tanh(x @ p["w1"] + p["b1"])
        x = jnp.tanh(x @ p["w2"] + p["b2"])
        logits = x @ p["w3"] + p["b3"]
    return jnp.exp(jnp.log(2.) * jnp.tanh(logits))
