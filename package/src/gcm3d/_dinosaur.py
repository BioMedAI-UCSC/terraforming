"""Single guarded entry point for the optional ``dinosaur`` (JAX) dependency.

Every dinosaur-dependent module in ``src.gcm3d`` imports the pieces it needs from
here, so there is exactly one place that raises the friendly "install the gcm3d
extra" message when the optional dependency is absent.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from dinosaur import (
        coordinate_systems,
        primitive_equations,
        primitive_equations_states,
        scales,
        sigma_coordinates,
        spherical_harmonic,
        time_integration,
        units,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "src.gcm3d's 3-D GCM core requires the optional 'gcm3d' extra "
        "(dinosaur + jax): pip install 'terraforming[gcm3d]'"
    ) from exc

__all__ = [
    "jax",
    "jnp",
    "coordinate_systems",
    "primitive_equations",
    "primitive_equations_states",
    "scales",
    "sigma_coordinates",
    "spherical_harmonic",
    "time_integration",
    "units",
]
