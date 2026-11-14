"""Celestial bodies and their planet-specific model configurations."""

from src.framework.intrinsic import IntrinsicParameters
from src.framework.orbital import OrbitalParameters
from src.framework.planet import Planet
from src.celestials.planets.mars import (
    MARS_AXIAL_TILT,
    MARS_DEFAULT_COMPOSITION,
    MARS_ECCENTRICITY,
    MARS_GRAVITY,
    MARS_MASS,
    MARS_ORBITAL_PERIOD,
    MARS_RADIUS,
    MARS_ROTATION_PERIOD,
    MARS_SEMI_MAJOR_AXIS,
    Mars,
)

__all__ = [
    "IntrinsicParameters",
    "MARS_AXIAL_TILT",
    "MARS_DEFAULT_COMPOSITION",
    "MARS_ECCENTRICITY",
    "MARS_GRAVITY",
    "MARS_MASS",
    "MARS_ORBITAL_PERIOD",
    "MARS_RADIUS",
    "MARS_ROTATION_PERIOD",
    "MARS_SEMI_MAJOR_AXIS",
    "Mars",
    "OrbitalParameters",
    "Planet",
]
