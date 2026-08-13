"""Mars — constants, and the PyTorch 0-D planet model built on them.

Two modules, split along a deliberate line:

``constants``
    Plain-``float`` physical constants (:class:`MarsConstants`, :data:`MARS`).
    Imports nothing but the standard library, so a JAX/NumPy backend can read
    the same numbers the torch model uses.

``planet``
    :class:`Mars`, the PyTorch box model, plus :func:`constants_to_tensors` —
    the single conversion point from those floats into ``torch`` scalars.

Import from here rather than reaching into the submodules::

    from src.celestials.planets.mars import Mars, MARS, MARS_ORBITAL_PERIOD
"""

from __future__ import annotations

from src.celestials.planets.mars.constants import (
    MARS,
    MARS_DEFAULT_COMPOSITION_PA,
    MarsConstants,
)
from src.celestials.planets.mars.planet import (
    MARS_AXIAL_TILT,
    MARS_CO2_FROST_POINT,
    MARS_CO2_LATENT_HEAT,
    MARS_DEFAULT_COMPOSITION,
    MARS_DIURNAL_SWING_AMP,
    MARS_ECCENTRICITY,
    MARS_GRAVITY,
    MARS_LS_PERIHELION,
    MARS_MASS,
    MARS_MAVEN_ESCAPE_RATE,
    MARS_ORBITAL_PERIOD,
    MARS_POLAR_CAP_FRACTION,
    MARS_RADIUS,
    MARS_ROTATION_PERIOD,
    MARS_SEMI_MAJOR_AXIS,
    MARS_SURFACE_EMISSIVITY,
    MARS_THERMAL_INERTIA,
    MARS_THERMAL_TIDE_PA,
    MARS_THERMAL_TIDE_PHASE,
    Mars,
    composition_to_tensors,
    constants_to_tensors,
)

__all__ = [
    # Framework-neutral constants
    "MARS",
    "MarsConstants",
    "MARS_DEFAULT_COMPOSITION_PA",
    # Model + conversion factories
    "Mars",
    "constants_to_tensors",
    "composition_to_tensors",
    # Legacy torch scalars
    "MARS_MASS",
    "MARS_RADIUS",
    "MARS_GRAVITY",
    "MARS_ROTATION_PERIOD",
    "MARS_SEMI_MAJOR_AXIS",
    "MARS_ECCENTRICITY",
    "MARS_ORBITAL_PERIOD",
    "MARS_AXIAL_TILT",
    "MARS_LS_PERIHELION",
    "MARS_SURFACE_EMISSIVITY",
    "MARS_THERMAL_INERTIA",
    "MARS_MAVEN_ESCAPE_RATE",
    "MARS_CO2_FROST_POINT",
    "MARS_CO2_LATENT_HEAT",
    "MARS_POLAR_CAP_FRACTION",
    "MARS_DIURNAL_SWING_AMP",
    "MARS_THERMAL_TIDE_PA",
    "MARS_THERMAL_TIDE_PHASE",
    "MARS_DEFAULT_COMPOSITION",
]
