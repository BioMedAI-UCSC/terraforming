"""gcm3d — a planet-agnostic differentiable 3-D GCM core on the NeuralGCM dycore.

A subpackage of the terraforming ``src`` package providing the mid-fidelity 3-D
tier of the model ladder. It is **not specific to any planet**: the core consumes
a :class:`~src.gcm3d.body.BodyConstants`, and concrete bodies supply an instance.
Mars's instance lives with the planet definition in
``src/celestials/planets/mars.py`` (``MARS_BODY_3D``), built from the existing
``MARS_*`` constants — so the core extends to other planets and moons by defining
a new ``BodyConstants``.

It depends on **JAX** (via ``dinosaur``), which the torch core does not — so
``dinosaur`` is an **optional extra**::

    pip install 'terraforming[gcm3d]'

``src.gcm3d.body`` (the ``BodyConstants`` abstraction) is pure Python and always
importable; the coordinate/equation builders require the extra. The two
frameworks meet only at the experiment layer via DLPack — never by cross-import.
See ``docs/ideas/dinosaur-mars-workplan.md`` for the phased plan.
"""

from __future__ import annotations

# Pure-Python body abstraction — always importable (no JAX / dinosaur).
from src.gcm3d.body import EARTH, BodyConstants

__all__ = ["BodyConstants", "EARTH", "__version__"]
__version__ = "0.0.1"

# Core builders need dinosaur (the gcm3d extra). Expose them when available;
# keep ``from src.gcm3d import BodyConstants`` working without the extra.
try:  # pragma: no cover - import-availability branch
    from src.gcm3d.coordinates import coordinate_system, grid
    from src.gcm3d.dynamics import (
        integrate,
        primitive_equations,
        reference_temperature,
        stepper,
    )
    from src.gcm3d.specs import nondimensionalization_scale, physics_specs

    __all__ += [
        "coordinate_system",
        "grid",
        "physics_specs",
        "nondimensionalization_scale",
        "primitive_equations",
        "reference_temperature",
        "stepper",
        "integrate",
    ]
except ModuleNotFoundError:
    pass
