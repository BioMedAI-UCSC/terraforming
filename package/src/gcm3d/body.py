"""Planet-agnostic celestial-body constants for the 3-D GCM core.

``BodyConstants`` is the abstraction that decouples the ``gcm3d`` core (built on
the NeuralGCM ``dinosaur`` dycore) from any particular planet. The core consumes
a ``BodyConstants``; concrete bodies (Mars, and later other planets/moons)
supply an instance from their own constants.

Mars's instance lives with the planet definition in
``src/celestials/planets/mars.py`` (built from the existing ``MARS_*`` constants,
so there is a single source of truth). This module stays free of any specific
body's values — only the schema, validation, and an Earth reference for tests.

Pure Python (no JAX/dinosaur): always importable, so the torch side of the
package can construct a ``BodyConstants`` without the optional ``gcm3d`` extra.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class BodyConstants:
    """Physical constants a planet/moon supplies to the 3-D GCM core.

    Exactly the quantities the dry dynamical core needs; body-specific physics
    (radiation, surface, volatiles) is added per body in later phases. All SI.
    Frozen so a body definition is immutable and hashable (JAX-static-friendly).

    Parameters
    ----------
    name : str
        Body name (e.g. "Mars", "Earth", "Titan").
    radius_m : float
        Mean radius (m).
    gravity_m_s2 : float
        Surface gravitational acceleration (m s^-2).
    rotation_period_s : float
        Sidereal rotation period (s); sets the angular velocity.
    gas_constant_j_kg_k : float
        Specific gas constant of the atmosphere, R = R_universal / M (J kg^-1 K^-1).
    cp_j_kg_k : float
        Isobaric specific heat capacity (J kg^-1 K^-1); with R sets kappa = R/cp.
    reference_temperature_k : float
        Reference temperature for the semi-implicit linearisation (K).
    reference_surface_pressure_pa : float
        Representative global-mean surface pressure (Pa).
    """

    name: str
    radius_m: float
    gravity_m_s2: float
    rotation_period_s: float
    gas_constant_j_kg_k: float
    cp_j_kg_k: float
    reference_temperature_k: float = 250.0
    reference_surface_pressure_pa: float = 1.0e5

    def __post_init__(self) -> None:
        for field_name in (
            "radius_m",
            "gravity_m_s2",
            "rotation_period_s",
            "gas_constant_j_kg_k",
            "cp_j_kg_k",
            "reference_temperature_k",
            "reference_surface_pressure_pa",
        ):
            value = getattr(self, field_name)
            if not (value > 0.0):
                raise ValueError(f"{field_name} must be positive, got {value!r}")

    @property
    def angular_velocity_s(self) -> float:
        """Rotation rate Omega = 2 pi / rotation_period (rad s^-1)."""
        return 2.0 * math.pi / self.rotation_period_s

    @property
    def kappa(self) -> float:
        """Poisson exponent kappa = R / cp (dimensionless)."""
        return self.gas_constant_j_kg_k / self.cp_j_kg_k

    @property
    def surface_area_m2(self) -> float:
        return 4.0 * math.pi * self.radius_m**2


# Earth reference — a second, well-known body. Used in tests to prove the core
# is planet-agnostic and to contrast Mars against known Earth values. Constants
# are standard dry-air / Earth values.
EARTH = BodyConstants(
    name="Earth",
    radius_m=6.371220e6,
    gravity_m_s2=9.80616,
    rotation_period_s=86_164.0905,  # sidereal day
    gas_constant_j_kg_k=287.04,  # dry air
    cp_j_kg_k=1004.6,  # dry air; kappa = R/cp ~ 2/7
    reference_temperature_k=288.0,
    reference_surface_pressure_pa=1.01325e5,
)
