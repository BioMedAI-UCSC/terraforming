"""Mars – concrete implementation of the abstract Planet (PyTorch backend).

Mars provides the **physics model** — the governing equations for how its
atmosphere, surface temperature, and ice budget evolve.  It does **not**
own the integration strategy; that belongs to the engine (TimeController).

Physics supplied to the engine:
    ``compute_derivatives(y)``   – coupled ODE RHS [dT/dt, dP/dt, dM_ice/dt]
    ``compute_fast_physics(dt)`` – reduced-order analytic/relaxation update

Physical constants are from the NASA Mars Fact Sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
"""

from __future__ import annotations

from typing import Dict, Optional

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

from src.framework.planet import (
    Planet,
)
from src.framework.atmosphere import Atmosphere
from src.framework.thermal import Thermal
from src.framework.water import Water
from src.framework.radiation import Radiation
from src.framework.magnetic import Magnetic
from src.framework.intrinsic import IntrinsicParameters
from src.framework.orbital import OrbitalParameters
from src.framework.atmosphere import Atmosphere
from src.framework.thermal import Thermal
from src.framework.water import Water
from src.framework.radiation import Radiation
from src.framework.magnetic import Magnetic

# Mars-specific constants  (all torch.Tensor, float64)
MARS_MASS: torch.Tensor             = _c(6.4171e23)             # kg
MARS_RADIUS: torch.Tensor           = _c(3.3895e6)              # m
MARS_GRAVITY: torch.Tensor          = _c(3.72076)               # m s⁻²
MARS_ROTATION_PERIOD: torch.Tensor  = _c(88_775.244)            # s  (1 sol)
MARS_SEMI_MAJOR_AXIS: torch.Tensor  = _c(2.27939200e11)         # m  (1.524 AU)
MARS_ECCENTRICITY: torch.Tensor     = _c(0.0934)                # dimensionless
MARS_ORBITAL_PERIOD: torch.Tensor   = _c(5.93568e7)             # s  (~687 d)
MARS_AXIAL_TILT: torch.Tensor       = _c(25.19 * 3.141592653589793 / 180.0)  # rad

# Default atmospheric composition (partial pressures in Pa, as torch.Tensor)
MARS_DEFAULT_COMPOSITION: Dict[str, torch.Tensor] = {
    "CO2": _c(580.0),
    "N2":  _c(15.0),
    "Ar":  _c(12.0),
    "O2":  _c(0.8),
    "CO":  _c(0.4),
}


class Mars(Planet):
    """Mars planetary model — state container + physics equations.

    Parameters
    ----------
    surface_temperature : float, optional
        Initial surface temperature (K).  Default 210 K.
    surface_pressure : float, optional
        Initial surface pressure (Pa).  Default 610 Pa.
    albedo : float, optional
        Bond albedo (0-1).  Default 0.25.
    greenhouse_factor : float, optional
        Effective greenhouse enhancement (≥1).  Default 1.02.
    composition : dict, optional
        Species → partial pressure (Pa).  Default is current Mars atmosphere.
    ice_mass : float, optional
        Initial polar + permafrost ice mass (kg).  Default 5 × 10¹⁵ kg.
    """

    def __init__(
        self,
        surface_temperature: float = 210.0,
        surface_pressure: float = 610.0,
        albedo: float = 0.25,
        # Mars's CO₂ atmosphere is very thin (610 Pa vs Earth's 101,325 Pa — 166× thinner). Earth's greenhouse factor is ~1.33 (288K actual vs 255K effective
        greenhouse_factor: float = 1.02,
        composition: Optional[Dict[str, float]] = None,
        ice_mass: float = 5.0e15,
        latitude: float = 22.0,
        longitude: float = 0.0,
    ) -> None:
        self.intrinsic_params = IntrinsicParameters(
            mass=MARS_MASS,
            radius=MARS_RADIUS,
            gravity=MARS_GRAVITY,
            rotation_period=MARS_ROTATION_PERIOD,
        )

        self.orbital_params = OrbitalParameters(
            semi_major_axis=MARS_SEMI_MAJOR_AXIS,
            eccentricity=MARS_ECCENTRICITY,
            orbital_period=MARS_ORBITAL_PERIOD,
            axial_tilt=MARS_AXIAL_TILT,
        )

        # Store initial-condition overrides (convert to tensors)
        self._init_temperature = _c(surface_temperature)
        self._init_pressure = _c(surface_pressure)
        self._init_albedo = _c(albedo)
        self._init_greenhouse = _c(greenhouse_factor)
        self._init_ice_mass = _c(ice_mass)
        self._init_latitude = _c(latitude) * PI / _c(180.0)
        self._init_longitude = _c(longitude) * PI / _c(180.0)

        # Composition: accept raw floats from user, convert to tensors
        if composition is not None:
            self._init_composition = {
                k: _c(v) for k, v in composition.items()
            }
        else:
            self._init_composition = {
                k: v.clone() for k, v in MARS_DEFAULT_COMPOSITION.items()
            }

        self.setup_properties()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    def setup_properties(self) -> None:
        """Initialize the planet's physical properties.

        Values sourced from NASA Mars Fact Sheet.
        """
        self.atmosphere = Atmosphere(
            surface_pressure=self._init_pressure.clone(),
            atmospheric_mass=_c(2.5e16),         # kg  (total atmosphere)
            composition=dict(self._init_composition),
        )
        self.thermal = Thermal(
            surface_temperature=self._init_temperature.clone(),
            greenhouse_factor=self._init_greenhouse.clone(),
        )
        self.water = Water(
            ice_mass=self._init_ice_mass.clone(),
            liquid_mass=_c(0.0),
            vapour_mass=_c(1.0e13),
        )
        self.radiation = Radiation(
            albedo=self._init_albedo.clone(),
            solar_flux=_c(0.0),
        )
        self.magnetic = Magnetic(
            magnetic_field_strength=_c(5.0e-9),  # T  (weak crustal remnants)
        )
        self.elapsed_time = _c(0.0)
        self.orbital_angle = _c(0.0)

    # ==================================================================
    # PHYSICS: Coupled ODE derivatives  (used by engine's RK4 integrator)
    # ==================================================================
    def compute_derivatives(self, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt for the coupled system y = [T, P, M_ice].

        Coupled ODE system:

        ┌──────────────────────────────────────────────────────────────┐
        │  dT/dt  = [ Q_in − Q_out ] / C                             │
        │         = [(1−α) F π R² − ε σ (T/f_gh)⁴ 4π R²] / C        │
        │                                                              │
        │  dP/dt  = −Ṁ_escape g / (4π R²)                            │
        │         where Ṁ_escape = 4π R² n(R) v_th exp(−λ)           │
        │         λ = G M m_CO2 / (k T R_exo)                         │
        │                                                              │
        │  dM_ice/dt = −(sublimation rate)                            │
        │            = −A_cap L_sub⁻¹ σ T⁴   (simplified)            │
        └──────────────────────────────────────────────────────────────┘

        References
        ----------
        Stefan-Boltzmann law : https://en.wikipedia.org/wiki/Stefan–Boltzmann_law
        Jeans escape         : https://en.wikipedia.org/wiki/Atmospheric_escape
        """
        s = self

        T = torch.maximum(y[0], _c(1.0))
        P = torch.maximum(y[1], _c(0.0))
        M_ice = torch.maximum(y[2], _c(0.0))

        # ==================================================================
        # --- dT/dt: energy balance ---
        # ==================================================================
        
        # We compute for a specific 1 km^2 patch (per unit area formulation)
        # Area A cancels out (1m^2 effective) for local patch heating/cooling.
        
        # Hour angle h. t=0 is midnight -> h = -pi
        omega = _c(2.0) * PI / self.intrinsic_params.rotation_period
        h = omega * s.elapsed_time - PI + self._init_longitude
        
        # Latitude
        lat = self._init_latitude
        
        # True anomaly is s.orbital_angle (where 0 is perihelion).
        # Mars perihelion is at Ls ~ 251 degrees.
        Ls_perihelion = _c(251.0) * PI / _c(180.0)
        Ls = s.orbital_angle + Ls_perihelion
        
        # Solar declination delta
        delta = torch.asin(torch.sin(self.orbital_params.axial_tilt) * torch.sin(Ls))
        
        # Diurnal incidence angle (cosine of zenith)
        cos_zenith = torch.maximum(
            _c(0.0),
            torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.cos(h)
        )
        
        # Q_in is Watts per m^2
        Q_in = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith

        emissivity = _c(0.95)
        T_eff = T / torch.maximum(s.thermal.greenhouse_factor, _c(1.0))
        
        # Q_out is Watts per m^2
        Q_out = emissivity * STEFAN_BOLTZMANN * T_eff ** 4

        # Specific heat capacity C (J / K / m^2).
        # Gale Crater (Curiosity) experiences higher diurnal swings. 
        # C = 1.0e5 produces the exact ~205K to ~270K profile seen in the REMS Sol 224 data.
        C_area = _c(1.0e5)

        dT_dt = (Q_in - Q_out) / C_area

        # ==================================================================
        # --- dM_ice/dt: Polar CO2 Sublimation & Condensation ---
        # ==================================================================
        
        # Using a phenomenological two-pole radiative balance instead of global T
        T_frost = _c(149.0)
        L_sub = _c(5.7e5) # Latent heat J/kg
        # Each polar cap covers ~3% of the planet's surface mathematically
        A_cap_pole = _c(0.03) * _c(4.0) * PI * self.intrinsic_params.radius ** 2
        
        # Insolation at poles
        cos_zenith_N = torch.maximum(_c(0.0), torch.sin(delta))
        cos_zenith_S = torch.maximum(_c(0.0), -torch.sin(delta))
        
        Q_in_N = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_N
        Q_in_S = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_S
        Q_out_pole = _c(0.95) * STEFAN_BOLTZMANN * T_frost ** 4
        
        net_sub_N = (Q_in_N - Q_out_pole) * A_cap_pole / L_sub
        net_sub_S = (Q_in_S - Q_out_pole) * A_cap_pole / L_sub
        
        dMice_dt = -(net_sub_N + net_sub_S) # negative means sublimating (ice drops)
        dMice_dt = torch.where(
            (M_ice <= _c(0.0)) & (dMice_dt < _c(0.0)),
            _c(0.0),
            dMice_dt
        )

        # ==================================================================
        # --- dP/dt: Jeans escape + mass exchange from ice caps ---
        # ==================================================================

        m_co2 = _c(44.0 * 1.66054e-27)            # kg
        R_exo = self.intrinsic_params.radius + _c(200_000.0)        # m
        lam = G_NEWTON * self.intrinsic_params.mass * m_co2 / (BOLTZMANN_K * T * R_exo)
        v_th = torch.sqrt(_c(2.0) * BOLTZMANN_K * T / m_co2)
        n_exo = P / (BOLTZMANN_K * T)
        escape_rate = (
            _c(4.0) * PI * R_exo ** 2
            * n_exo * m_co2 * v_th
            * torch.exp(-lam)
        )
        
        A_planet = _c(4.0) * PI * self.intrinsic_params.radius ** 2
        
        # Loss to space is slow, but sublimation is fast
        dP_dt = (-escape_rate * self.intrinsic_params.gravity / A_planet) + (-dMice_dt * self.intrinsic_params.gravity / A_planet)

        return torch.stack([dT_dt, dP_dt, dMice_dt])

    # ==================================================================
    # PHYSICS: Reduced-order analytic update  (used by engine's fast path)
    # ==================================================================
    def compute_fast_physics(self, dt: torch.Tensor) -> None:
        """Apply reduced-order physics to the planet.

        Assumes ``solar_flux`` is already current
        (the engine calls ``advance_orbit`` beforehand).

        Strategy:
        1. Compute radiative-equilibrium temperature analytically:
               T_eq = [ (1−α) F / (4 ε σ) ]^(1/4)  ×  f_gh

        2. Relax surface temperature toward T_eq exponentially:
               T(t+dt) = T_eq + (T − T_eq) exp(−dt / τ)
           where τ is the thermal inertia timescale.

        3. Update pressure via a first-order Euler escape term.

        4. Update ice mass via first-order sublimation.

        References
        ----------
        Energy balance : https://scied.ucar.edu/learning-zone/how-climate-works/energy-balance
        Relaxation     : "Newtonian cooling" approximation
        """
        dt = torch.as_tensor(dt, dtype=TF_DTYPE)
        s = self

        # --- Step 1: Equilibrium temperature (Latitude-aware) ---
        Ls_perihelion = _c(251.0) * PI / _c(180.0)
        Ls = s.orbital_angle + Ls_perihelion
        delta = torch.asin(torch.sin(self.orbital_params.axial_tilt) * torch.sin(Ls))
        lat = self._init_latitude

        # Mean diurnal insolation factor (daily average of cos_zenith)
        # cos(h0) = -tan(lat) * tan(delta)
        cos_h0 = torch.clamp(-torch.tan(lat) * torch.tan(delta), -1.0, 1.0)
        h0 = torch.acos(cos_h0)
        insolation_factor = (h0 * torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.sin(h0)) / PI
        insolation_factor = torch.maximum(insolation_factor, _c(0.0))

        emissivity = _c(0.95)
        absorbed = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * insolation_factor
        # Denominator is just emissivity*sigma because insolation_factor already averages the source
        T_eq_base = (absorbed / (emissivity * STEFAN_BOLTZMANN)) ** 0.25
        T_eq = T_eq_base * s.thermal.greenhouse_factor
        
        # Superimpose diurnal variation onto equilibrium temperature
        # Amplitude decreases at high latitudes near winter
        swing = _c(50.0) * torch.cos(lat)
        omega = _c(2.0) * PI / self.intrinsic_params.rotation_period
        T_eq = T_eq - swing * torch.cos(omega * s.elapsed_time + self._init_longitude)

        # --- Step 2: Exponential relaxation ---
        T_cur = torch.maximum(s.thermal.surface_temperature, _c(1.0))
        # Updated thermal inertia characteristic tau consistent with ODE formulation
        tau = _c(1.0e5) / (_c(4.0) * emissivity * STEFAN_BOLTZMANN * T_cur ** 3)
        tau = torch.maximum(tau, _c(1.0))
        s.thermal.surface_temperature = T_eq + (T_cur - T_eq) * torch.exp(-dt / tau)
        s.thermal.surface_temperature = torch.maximum(s.thermal.surface_temperature, _c(1.0))

        # --- Step 3: Ice budget (Polar CO2 Sublimation & Condensation) ---
        # Recalculate Ls and delta since they are local to the function
        Ls_perihelion = _c(251.0) * PI / _c(180.0)
        Ls = s.orbital_angle + Ls_perihelion
        delta = torch.asin(torch.sin(self.orbital_params.axial_tilt) * torch.sin(Ls))
        
        T_frost = _c(149.0)
        L_sub = _c(5.7e5)
        A_cap_pole = _c(0.03) * _c(4.0) * PI * self.intrinsic_params.radius ** 2
        
        cos_zenith_N = torch.maximum(_c(0.0), torch.sin(delta))
        cos_zenith_S = torch.maximum(_c(0.0), -torch.sin(delta))
        
        Q_in_N = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_N
        Q_in_S = (_c(1.0) - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_S
        Q_out_pole = _c(0.95) * STEFAN_BOLTZMANN * T_frost ** 4
        
        net_sub_N = (Q_in_N - Q_out_pole) * A_cap_pole / L_sub
        net_sub_S = (Q_in_S - Q_out_pole) * A_cap_pole / L_sub
        
        dMice = -(net_sub_N + net_sub_S) * dt
        dMice = torch.where(
            (s.water.ice_mass <= _c(0.0)) & (dMice < _c(0.0)),
            _c(0.0),
            dMice
        )
        s.water.ice_mass = torch.maximum(s.water.ice_mass + dMice, _c(0.0))

        # --- Step 4: Pressure (Jeans escape + Mass exchange) ---
        m_co2 = _c(44.0 * 1.66054e-27)
        R_exo = self.intrinsic_params.radius + _c(200_000.0)
        T = s.thermal.surface_temperature
        lam = G_NEWTON * self.intrinsic_params.mass * m_co2 / (BOLTZMANN_K * T * R_exo)
        v_th = torch.sqrt(_c(2.0) * BOLTZMANN_K * T / m_co2)
        n_exo = s.atmosphere.surface_pressure / (BOLTZMANN_K * T)
        escape_rate = (
            _c(4.0) * PI * R_exo ** 2 * n_exo * m_co2 * v_th
            * torch.exp(-lam)
        )
        A_planet = _c(4.0) * PI * self.intrinsic_params.radius ** 2
        dP_escape = -escape_rate * self.intrinsic_params.gravity / A_planet * dt
        dP_sublimation = -dMice * self.intrinsic_params.gravity / A_planet
        
        s.atmosphere.surface_pressure = torch.maximum(s.atmosphere.surface_pressure + dP_escape + dP_sublimation, _c(0.0))
