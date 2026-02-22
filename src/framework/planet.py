"""Abstract Planet framework — PyTorch backend.

Defines the abstract base class ``Planet``, the ``PlanetaryState`` snapshot,
and ``OrbitalParameters``.  Concrete planets (e.g. Mars) inherit from
``Planet`` and provide planet-specific **physics** (derivatives, equilibrium
calculations), while the integration strategy (RK4, reduced-order) lives in
the engine (``TimeController``).

Separation of concerns:
    Planet       → *what* the physics equations are  (state + derivatives)
    Engine       → *how* those equations are integrated  (RK4 / relaxation)

All numerical values are stored as ``torch.Tensor`` scalars (dtype=torch.float64).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import torch

from src.constants import (
    TF_DTYPE,
    _c,
    AU_METRES,
    BOLTZMANN_K,
    G_NEWTON,
    PI,
    SOLAR_CONSTANT_1AU,
    STEFAN_BOLTZMANN,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OrbitalParameters:
    """Keplerian orbital elements (assumed constant over short runs)."""

    semi_major_axis: torch.Tensor      # metres
    eccentricity: torch.Tensor         # dimensionless, 0 < e < 1
    orbital_period: torch.Tensor       # seconds
    axial_tilt: torch.Tensor           # radians

    def distance_from_sun(self, theta: torch.Tensor) -> torch.Tensor:
        """Heliocentric distance at true anomaly *theta* (radians).

        Equation (Kepler's first law – conic section):
            r(θ) = a(1 − e²) / (1 + e cos θ)

        Reference: https://en.wikipedia.org/wiki/Kepler_orbit
        """
        return (
            self.semi_major_axis
            * (_c(1.0) - self.eccentricity ** 2)
            / (_c(1.0) + self.eccentricity * torch.cos(theta))
        )


@dataclass
class PlanetaryState:
    """Complete mutable snapshot of a planet at a single instant.

    Groups:
        Atmospheric  – pressure, temperature, composition
        Thermal      – surface temperature, greenhouse factor
        Water/Ice    – ice / liquid / vapour masses
        Radiation    – albedo, solar flux
        Magnetic     – surface field strength

    All scalar fields are ``torch.Tensor`` with dtype ``torch.float64``.
    """

    # --- Atmospheric ---
    surface_pressure: torch.Tensor = field(       # Pa
        default_factory=lambda: _c(0.0))
    atmospheric_mass: torch.Tensor = field(       # kg
        default_factory=lambda: _c(0.0))
    composition: Dict[str, torch.Tensor] = field( # species → partial pressure (Pa)
        default_factory=dict,
    )

    # --- Thermal ---
    surface_temperature: torch.Tensor = field(    # K
        default_factory=lambda: _c(0.0))
    greenhouse_factor: torch.Tensor = field(      # dimensionless (≥ 1)
        default_factory=lambda: _c(1.0))

    # --- Water budget ---
    ice_mass: torch.Tensor = field(               # kg
        default_factory=lambda: _c(0.0))
    liquid_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))
    vapour_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))

    # --- Radiation ---
    albedo: torch.Tensor = field(                 # 0-1
        default_factory=lambda: _c(0.25))
    solar_flux: torch.Tensor = field(             # W m⁻² (at current distance)
        default_factory=lambda: _c(0.0))

    # --- Magnetic ---
    magnetic_field_strength: torch.Tensor = field(  # Tesla at surface
        default_factory=lambda: _c(0.0))

    # --- Time tracking (set by engine) ---
    elapsed_time: torch.Tensor = field(           # total sim seconds since epoch
        default_factory=lambda: _c(0.0))
    orbital_angle: torch.Tensor = field(          # radians (true anomaly, 0-2π)
        default_factory=lambda: _c(0.0))

    def copy(self) -> PlanetaryState:
        """Deep copy of the entire state."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Abstract Planet  —  state container + physics model
# ---------------------------------------------------------------------------
class Planet(ABC):
    """Abstract base class for planetary bodies.

    A ``Planet`` is a **state container** combined with a **physics model**.
    It knows *what* the governing equations are (derivatives, equilibrium)
    but does **not** know *how* to integrate them — that is the engine's job.

    Sub-classes **must** implement:
        ``initialize_state()``       – return a PlanetaryState with physical ICs
        ``compute_derivatives(y)``   – ODE RHS for the coupled system
        ``compute_fast_physics(dt)`` – reduced-order analytic update to state
    """

    # --- Concrete attributes set by subclass __init__ ---
    mass: torch.Tensor                  # kg
    radius: torch.Tensor                # m
    rotation_period: torch.Tensor       # seconds (1 sidereal day)
    gravity: torch.Tensor               # m s⁻² (surface)
    orbital_params: OrbitalParameters
    state: PlanetaryState

    # ------------------------------------------------------------------
    # Abstract interface  —  physics provided by each planet
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize_state(self) -> PlanetaryState:
        """Return a fresh PlanetaryState with physically-motivated ICs."""
        ...

    @abstractmethod
    def compute_derivatives(self, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt for the coupled ODE system.

        Parameters
        ----------
        y : torch.Tensor, shape [3]
            State vector ``[T_surface, P_surface, M_ice]``.

        Returns
        -------
        torch.Tensor, shape [3]
            Time derivatives ``[dT/dt, dP/dt, dM_ice/dt]``.

        The engine calls this with intermediate state vectors during
        RK4 sub-steps.  Implementations should use ``self.state`` only
        for quantities that are *not* part of *y* (e.g. albedo,
        greenhouse factor, solar flux).
        """
        ...

    @abstractmethod
    def compute_fast_physics(self, dt: torch.Tensor) -> None:
        """Apply a single reduced-order physics step to ``self.state``.

        This method computes analytic / linearised approximations
        (equilibrium temperature, exponential relaxation, first-order
        Euler updates for pressure and ice) and writes the results
        directly into ``self.state``.

        The engine is responsible for calling ``advance_orbit`` before
        this method, so ``self.state.solar_flux`` is already up-to-date.

        Parameters
        ----------
        dt : torch.Tensor
            Timestep in seconds.
        """
        ...

    # ------------------------------------------------------------------
    # State packing / unpacking  (used by the engine's ODE integrators)
    # ------------------------------------------------------------------
    def pack_state(self) -> torch.Tensor:
        """Pack the evolvable variables into a 1-D tensor [T, P, M_ice]."""
        s = self.state
        return torch.stack([
            s.surface_temperature,
            s.surface_pressure,
            s.ice_mass,
        ])

    def unpack_state(self, y: torch.Tensor) -> None:
        """Unpack a 1-D tensor [T, P, M_ice] back into ``self.state``.

        Values are clamped to physical bounds.
        """
        s = self.state
        s.surface_temperature = torch.maximum(y[0], _c(1.0))
        s.surface_pressure    = torch.maximum(y[1], _c(0.0))
        s.ice_mass            = torch.maximum(y[2], _c(0.0))

    # ------------------------------------------------------------------
    # Shared helpers available to every planet
    # ------------------------------------------------------------------
    def solar_flux_at_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Solar irradiance at heliocentric *distance* (metres).

        Equation (inverse-square law):
            F(d) = F₀ × (1 AU / d)²

        Reference: https://en.wikipedia.org/wiki/Solar_irradiance
        """
        return SOLAR_CONSTANT_1AU * (AU_METRES / distance) ** 2

    def advance_orbit(self, dt: torch.Tensor) -> None:
        """Advance orbital angle (mean-motion approximation) and update
        solar flux in ``self.state``.

        Called by the engine at each timestep *before* physics updates.

        Equations:
            dθ/dt = 2π / T_orbital          (mean motion)
            r(θ)  = a(1−e²)/(1+e cos θ)    (Kepler ellipse)
            F     = F₀ (1 AU / r)²          (inverse-square)
        """
        dt = torch.as_tensor(dt, dtype=TF_DTYPE)
        s = self.state
        s.elapsed_time = s.elapsed_time + dt
        s.orbital_angle = torch.remainder(
            s.orbital_angle
            + _c(2.0) * PI * dt / self.orbital_params.orbital_period,
            _c(2.0) * PI,
        )

        distance = self.orbital_params.distance_from_sun(s.orbital_angle)
        s.solar_flux = self.solar_flux_at_distance(distance)
