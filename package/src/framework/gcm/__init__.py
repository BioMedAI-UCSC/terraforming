"""Planet-independent GCM infrastructure and Dinosaur integration."""

from src.framework.gcm.body import BodyConstants, EARTH

__all__ = ["BodyConstants", "EARTH"]

try:  # Keep BodyConstants importable without the optional JAX/Dinosaur extra.
    from src.framework.gcm.dynamics import (
        integrate,
        primitive_equations,
        reference_temperature,
        stepper,
    )
    from src.framework.gcm.specs import nondimensionalization_scale, physics_specs

    __all__ += [
        "integrate",
        "nondimensionalization_scale",
        "physics_specs",
        "primitive_equations",
        "reference_temperature",
        "stepper",
    ]
except ModuleNotFoundError:
    pass
