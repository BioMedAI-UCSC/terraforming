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

import math
from abc import ABC, abstractmethod

import torch

from src.constants import TF_DTYPE

from src.framework.atmosphere import Atmosphere
from src.framework.thermal import Thermal
from src.framework.water import Water
from src.framework.radiation import Radiation
from src.framework.magnetic import Magnetic
from src.framework.intrinsic import IntrinsicParameters
from src.framework.orbital import OrbitalParameters

# Python-float constants — device-agnostic (PyTorch broadcasts scalars to any device)
_SOLAR_CONST_1AU = 1361.0           # W m⁻²
_AU_METRES       = 1.49597870700e11  # m
_TWO_PI          = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Abstract Planet  —  physics model & property container
# ---------------------------------------------------------------------------
class Planet(ABC):
    """Abstract base class for planetary bodies.

    A ``Planet`` is a **property container** combined with a **physics model**.
    It knows *what* the governing equations are (derivatives, equilibrium)
    but does **not** know *how* to integrate them — that is the engine's job.

    Sub-classes **must** implement:
        ``setup_properties()``       – initialize the planet's property dataclasses
        ``compute_derivatives(y)``   – ODE RHS for the coupled system
        ``compute_fast_physics(dt)`` – reduced-order analytic update
    """

    # --- Concrete attributes set by subclass __init__ ---
    intrinsic_params: IntrinsicParameters
    orbital_params: OrbitalParameters
    _device: torch.device          # set by subclass before setup_properties()

    # --- Planetary Properties ---
    atmosphere: Atmosphere
    thermal: Thermal
    water: Water
    radiation: Radiation
    magnetic: Magnetic

    # --- Time tracking ---
    elapsed_time: torch.Tensor
    orbital_angle: torch.Tensor

    # ------------------------------------------------------------------
    # Abstract interface  —  physics provided by each planet
    # ------------------------------------------------------------------
    @abstractmethod
    def setup_properties(self) -> None:
        """Initialize the planet's physical properties."""
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
        RK4 sub-steps.  Implementations should use ``self`` only
        for quantities that are *not* part of *y* (e.g. albedo,
        greenhouse factor, solar flux).
        """
        ...

    @abstractmethod
    def compute_fast_physics(self, dt: torch.Tensor) -> None:
        """Apply a single reduced-order physics step to the planet.

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
        return torch.stack([
            self.thermal.surface_temperature,
            self.atmosphere.surface_pressure,
            self.water.ice_mass,
        ])

    def unpack_state(self, y: torch.Tensor) -> None:
        """Unpack a 1-D tensor [T, P, M_ice] back into the planet.

        Values are clamped to physical bounds.  Uses ``torch.where`` instead
        of a Python ``if`` so this method is device-agnostic and compatible
        with batched / compiled execution.
        """
        self.thermal.surface_temperature = y[0].clamp(min=1.0)
        self.atmosphere.surface_pressure = y[1].clamp(min=0.0)
        new_total = y[2].clamp(min=0.0)

        old_total = self.water.ice_mass_north + self.water.ice_mass_south
        # Apportion change between north/south reservoirs proportionally.
        # Use torch.where to avoid a Python boolean check on a GPU tensor.
        safe_total = old_total.clamp(min=1e-30)
        f_n = torch.where(
            old_total > 0.0,
            self.water.ice_mass_north / safe_total,
            old_total.new_full((), 0.5),  # same device & dtype, scalar shape
        )
        f_s = 1.0 - f_n
        self.water.ice_mass_north = (new_total * f_n).clamp(min=0.0)
        self.water.ice_mass_south = (new_total * f_s).clamp(min=0.0)
        self.water.ice_mass       = new_total

    # ------------------------------------------------------------------
    # Shared helpers available to every planet
    # ------------------------------------------------------------------
    def solar_flux_at_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Solar irradiance at heliocentric *distance* (metres).

        Uses Python-float constants so the result lives on whatever device
        ``distance`` is on — no explicit ``.to(device)`` required.

        Equation (inverse-square law):
            F(d) = F₀ × (1 AU / d)²

        Reference:
            Kopp & Lean (2011), "A new, lower value of total solar irradiance:
            Evidence and climate significance", Geophysical Research Letters.
            https://doi.org/10.1029/2010GL045777
        """
        return _SOLAR_CONST_1AU * (_AU_METRES / distance) ** 2


    def advance_orbit(self, dt: torch.Tensor) -> None:
        """Advance the orbit using Keplerian timing and update solar flux.

        ``orbital_angle`` is the true anomaly ν measured from perihelion.
        Mean anomaly M advances uniformly in time. Kepler's equation is then
        solved to recover eccentric anomaly E, which is converted back to true
        anomaly ν for distance and solar-flux calculation.

        Equations:
            M = E - e sin(E)                 (Kepler's equation)
            dM/dt = 2π / T_orbital           (mean motion)
            r(ν) = a(1−e²)/(1+e cos ν)       (Kepler ellipse), (ν measured from perihelion)
            F = F₀ (1 AU / r)²               (inverse-square flux)
        """
        self.elapsed_time = self.elapsed_time + dt

        e = self.orbital_params.eccentricity
        nu = self.orbital_angle

        # True anomaly ν -> eccentric anomaly E
        eccentric_anomaly = 2.0 * torch.atan2(
            torch.sqrt(1.0 - e) * torch.sin(nu / 2.0),
            torch.sqrt(1.0 + e) * torch.cos(nu / 2.0),
        )

        # Eccentric anomaly E -> mean anomaly M.
        # M = E - e sin(E)
        mean_anomaly = eccentric_anomaly - e * torch.sin(eccentric_anomaly)

        # Mean anomaly advances uniformly with time.
        # Advances mean anomaly by the mean motion (2π / T_orbital) scaled by dt.
        mean_anomaly = torch.remainder(
            mean_anomaly
            + _TWO_PI * dt / self.orbital_params.orbital_period,
            _TWO_PI,
        )

        # Solve Kepler's equation: M = E - e sin(E).
        # Fixed iteration count preserves tensor/device compatibility.
        eccentric_anomaly = mean_anomaly + e * torch.sin(mean_anomaly)
        for _ in range(8):
            residual = (
                eccentric_anomaly
                - e * torch.sin(eccentric_anomaly)
                - mean_anomaly
            )
            derivative = 1.0 - e * torch.cos(eccentric_anomaly)
            eccentric_anomaly = eccentric_anomaly - residual / derivative

        # Eccentric anomaly E -> true anomaly ν.
        true_anomaly = 2.0 * torch.atan2(
            torch.sqrt(1.0 + e) * torch.sin(eccentric_anomaly / 2.0),
            torch.sqrt(1.0 - e) * torch.cos(eccentric_anomaly / 2.0),
        )
        self.orbital_angle = torch.remainder(true_anomaly, _TWO_PI)

        distance = self.orbital_params.distance_from_sun(self.orbital_angle)
        self.radiation.solar_flux = self.solar_flux_at_distance(distance)