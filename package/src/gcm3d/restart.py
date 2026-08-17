"""Versioned restart files and spin-up/averaging windows for gcm3d.

Restart files use NumPy NPZ plus JSON metadata, not pickle. They are intentionally
limited to the stable :class:`ColumnPhysicsState` contract so incompatible changes
fail explicitly instead of silently restoring a malformed JAX pytree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.gcm3d._dinosaur import jax, jnp, primitive_equations
from src.gcm3d.dynamics import integrate
from src.gcm3d.physics import ColumnPhysicsState

RESTART_FORMAT_VERSION = 2


def save_restart(state: ColumnPhysicsState, path) -> Path:
    """Save every prognostic field required for a bitwise-continuous restart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dyn = state.dynamics
    tracer_names = sorted(dyn.tracers)
    metadata = {
        "format_version": RESTART_FORMAT_VERSION,
        "tracer_names": tracer_names,
        "sim_time_is_none": dyn.sim_time is None,
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "vorticity": np.asarray(dyn.vorticity),
        "divergence": np.asarray(dyn.divergence),
        "temperature_variation": np.asarray(dyn.temperature_variation),
        "log_surface_pressure": np.asarray(dyn.log_surface_pressure),
        "surface_temperature": np.asarray(state.surface_temperature),
        "co2_ice": np.asarray(state.co2_ice),
        "ground_temperature": np.asarray(state.ground_temperature),
    }
    if dyn.sim_time is not None:
        arrays["sim_time"] = np.asarray(dyn.sim_time)
    for index, name in enumerate(tracer_names):
        arrays[f"tracer_{index}"] = np.asarray(dyn.tracers[name])
    np.savez_compressed(path, **arrays)
    return path


def load_restart(path) -> ColumnPhysicsState:
    """Load a restart, rejecting unknown versions or incomplete state."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        version = metadata.get("format_version")
        if version != RESTART_FORMAT_VERSION:
            raise ValueError(
                f"unsupported gcm3d restart version {version!r}; "
                f"expected {RESTART_FORMAT_VERSION}"
            )
        tracers = {
            name: jnp.asarray(data[f"tracer_{index}"])
            for index, name in enumerate(metadata["tracer_names"])
        }
        sim_time = None if metadata["sim_time_is_none"] else jnp.asarray(data["sim_time"])
        dynamics = primitive_equations.State(
            vorticity=jnp.asarray(data["vorticity"]),
            divergence=jnp.asarray(data["divergence"]),
            temperature_variation=jnp.asarray(data["temperature_variation"]),
            log_surface_pressure=jnp.asarray(data["log_surface_pressure"]),
            tracers=tracers,
            sim_time=sim_time,
        )
        return ColumnPhysicsState(
            dynamics,
            jnp.asarray(data["surface_temperature"]),
            jnp.asarray(data["co2_ice"]),
            jnp.asarray(data["ground_temperature"]),
        )


def integrate_with_averaging(
    step_fn,
    initial_state,
    spinup_steps: int,
    average_steps: int,
    sample_every: int = 1,
    diagnostic_fn=lambda state: state,
):
    """Run spin-up, then return final state and a sampled diagnostic time mean."""
    if spinup_steps < 0:
        raise ValueError(f"spinup_steps must be >= 0, got {spinup_steps}")
    if average_steps < 1:
        raise ValueError(f"average_steps must be >= 1, got {average_steps}")
    if sample_every < 1 or average_steps % sample_every:
        raise ValueError("sample_every must be positive and divide average_steps")

    state = (integrate(step_fn, initial_state, spinup_steps)
             if spinup_steps else initial_state)
    n_samples = average_steps // sample_every

    def sample_block(carry, _):
        evolved = integrate(step_fn, carry, sample_every)
        return evolved, diagnostic_fn(evolved)

    final, samples = jax.lax.scan(sample_block, state, None, length=n_samples)
    mean = jax.tree_util.tree_map(lambda values: jnp.mean(values, axis=0), samples)
    return final, mean
