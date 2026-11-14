"""Parameterized, differentiable integration without reporting or host I/O."""
from typing import NamedTuple
import math

from src.framework.gcm._dinosaur import jax, jnp
from src.framework.gcm.dynamics import stepper


class Rollout(NamedTuple):
    final_state: object
    trajectory: object
    steps: object


def make_parameterized_step(equation_fn, dt_seconds, specs):
    """Bind static discretization to ``equation_fn(params)``.

    Equation construction must be trace-safe. Bind learned parameters to a
    radiation callable inside equation_fn; never convert them to host arrays.
    """
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")

    def step(params, state):
        return stepper(equation_fn(params), dt_seconds, specs)(state)
    return step


def rollout(step_fn, params, initial_state, n_steps, save_every=None, *, remat=False):
    """Advance ``step_fn(params, state)`` with optional sampled trajectory.

    Static integer counts are required. Samples exclude the initial state and
    include the final state even when n_steps is not divisible by save_every.
    With save_every=None, trajectory is None and no states are stacked. remat
    checkpoints each step for reverse-mode memory/computation tradeoffs. Batch
    externally with jax.vmap, sharing params and mapping only initial_state.
    """
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive static integer")
    if save_every is not None and (not isinstance(save_every, int) or save_every < 1):
        raise ValueError("save_every must be a positive static integer or None")
    advance = jax.checkpoint(step_fn) if remat else step_fn

    def body(state, _):
        return advance(params, state), None

    if save_every is None:
        final, _ = jax.lax.scan(body, initial_state, None, length=n_steps)
        return Rollout(final, None, jnp.empty((0,), dtype=jnp.int32))

    count, remainder = divmod(n_steps, save_every)

    def chunk(state, _):
        final, _ = jax.lax.scan(body, state, None, length=save_every)
        return final, final

    final, samples = jax.lax.scan(chunk, initial_state, None, length=count)
    steps = jnp.arange(1, count + 1) * save_every
    if remainder:
        final, _ = jax.lax.scan(body, final, None, length=remainder)
        samples = jax.tree.map(lambda a, b: jnp.concatenate([a, b[None]], axis=0), samples, final)
        steps = jnp.concatenate([steps, jnp.asarray([n_steps])])
    return Rollout(final, samples, steps)
