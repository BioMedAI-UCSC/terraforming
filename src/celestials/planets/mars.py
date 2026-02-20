"""Mars – concrete implementation of the abstract Planet (TensorFlow backend).

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

import tensorflow as tf

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

from src.celestials.framework.planet import (
    OrbitalParameters,
    Planet,
    PlanetaryState,
)

# Mars-specific constants  (all tf.Tensor, float64)
MARS_MASS: tf.Tensor             = _c(6.4171e23)             # kg
MARS_RADIUS: tf.Tensor           = _c(3.3895e6)              # m
MARS_GRAVITY: tf.Tensor          = _c(3.72076)               # m s⁻²
MARS_ROTATION_PERIOD: tf.Tensor  = _c(88_775.244)            # s  (1 sol)
MARS_SEMI_MAJOR_AXIS: tf.Tensor  = _c(2.27939200e11)         # m  (1.524 AU)
MARS_ECCENTRICITY: tf.Tensor     = _c(0.0934)                # dimensionless
MARS_ORBITAL_PERIOD: tf.Tensor   = _c(5.93568e7)             # s  (~687 d)
MARS_AXIAL_TILT: tf.Tensor       = _c(25.19 * 3.141592653589793 / 180.0)  # rad

# Default atmospheric composition (partial pressures in Pa, as tf.Tensor)
MARS_DEFAULT_COMPOSITION: Dict[str, tf.Tensor] = {
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
        greenhouse_factor: float = 1.02,
        composition: Optional[Dict[str, float]] = None,
        ice_mass: float = 5.0e15,
    ) -> None:
        self.mass = MARS_MASS
        self.radius = MARS_RADIUS
        self.gravity = MARS_GRAVITY
        self.rotation_period = MARS_ROTATION_PERIOD

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

        # Composition: accept raw floats from user, convert to tensors
        if composition is not None:
            self._init_composition = {
                k: _c(v) for k, v in composition.items()
            }
        else:
            self._init_composition = {
                k: tf.identity(v) for k, v in MARS_DEFAULT_COMPOSITION.items()
            }

        self.state = self.initialize_state()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    def initialize_state(self) -> PlanetaryState:
        """Create PlanetaryState with Mars-specific initial conditions.

        Values sourced from NASA Mars Fact Sheet.
        """
        return PlanetaryState(
            surface_pressure=tf.identity(self._init_pressure),
            atmospheric_mass=_c(2.5e16),         # kg  (total atmosphere)
            composition=dict(self._init_composition),
            surface_temperature=tf.identity(self._init_temperature),
            greenhouse_factor=tf.identity(self._init_greenhouse),
            ice_mass=tf.identity(self._init_ice_mass),
            liquid_mass=_c(0.0),
            vapour_mass=_c(1.0e13),
            albedo=tf.identity(self._init_albedo),
            solar_flux=_c(0.0),
            magnetic_field_strength=_c(5.0e-9),  # T  (weak crustal remnants)
            elapsed_time=_c(0.0),
            orbital_angle=_c(0.0),
        )

    # ==================================================================
    # PHYSICS: Coupled ODE derivatives  (used by engine's RK4 integrator)
    # ==================================================================
    def compute_derivatives(self, y: tf.Tensor) -> tf.Tensor:
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
        s = self.state

        T = tf.maximum(y[0], _c(1.0))
        P = tf.maximum(y[1], _c(0.0))
        M_ice = tf.maximum(y[2], _c(0.0))

        # --- dT/dt: energy balance ---
        Q_in = (_c(1.0) - s.albedo) * s.solar_flux * PI * self.radius ** 2

        emissivity = _c(0.95)
        T_eff = T / tf.maximum(s.greenhouse_factor, _c(1.0))
        Q_out = (
            emissivity
            * STEFAN_BOLTZMANN
            * T_eff ** 4
            * _c(4.0) * PI * self.radius ** 2
        )

        C = _c(2.0e6) * _c(4.0) * PI * self.radius ** 2   # J K⁻¹

        dT_dt = (Q_in - Q_out) / C

        # --- dP/dt: Jeans escape ---
        m_co2 = _c(44.0 * 1.66054e-27)            # kg
        R_exo = self.radius + _c(200_000.0)        # m
        lam = G_NEWTON * self.mass * m_co2 / (BOLTZMANN_K * T * R_exo)
        v_th = tf.math.sqrt(_c(2.0) * BOLTZMANN_K * T / m_co2)
        n_exo = P / (BOLTZMANN_K * T)
        escape_rate = (
            _c(4.0) * PI * R_exo ** 2
            * n_exo * m_co2 * v_th
            * tf.math.exp(-lam)
        )
        dP_dt = -escape_rate * self.gravity / (
            _c(4.0) * PI * self.radius ** 2
        )

        # --- dM_ice/dt: sublimation ---
        L_sub = _c(5.7e5)
        A_cap = _c(0.05) * _c(4.0) * PI * self.radius ** 2
        sublimation_flux = STEFAN_BOLTZMANN * T ** 4
        dMice_dt = tf.where(
            M_ice > _c(0.0),
            -A_cap * sublimation_flux / L_sub,
            _c(0.0),
        )

        return tf.stack([dT_dt, dP_dt, dMice_dt])

    # ==================================================================
    # PHYSICS: Reduced-order analytic update  (used by engine's fast path)
    # ==================================================================
    def compute_fast_physics(self, dt: tf.Tensor) -> None:
        """Apply reduced-order physics to ``self.state``.

        Assumes ``self.state.solar_flux`` is already current
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
        dt = tf.cast(dt, TF_DTYPE)
        s = self.state

        # --- Step 1: Equilibrium temperature ---
        emissivity = _c(0.95)
        absorbed = (_c(1.0) - s.albedo) * s.solar_flux
        T_eq_base = (absorbed / (_c(4.0) * emissivity * STEFAN_BOLTZMANN)) ** 0.25
        T_eq = T_eq_base * s.greenhouse_factor

        # --- Step 2: Exponential relaxation ---
        T_cur = tf.maximum(s.surface_temperature, _c(1.0))
        tau = _c(2.0e6) / (_c(4.0) * emissivity * STEFAN_BOLTZMANN * T_cur ** 3)
        tau = tf.maximum(tau, _c(1.0))
        s.surface_temperature = T_eq + (T_cur - T_eq) * tf.math.exp(-dt / tau)
        s.surface_temperature = tf.maximum(s.surface_temperature, _c(1.0))

        # --- Step 3: Pressure (first-order Jeans escape) ---
        m_co2 = _c(44.0 * 1.66054e-27)
        R_exo = self.radius + _c(200_000.0)
        T = s.surface_temperature
        lam = G_NEWTON * self.mass * m_co2 / (BOLTZMANN_K * T * R_exo)
        v_th = tf.math.sqrt(_c(2.0) * BOLTZMANN_K * T / m_co2)
        n_exo = s.surface_pressure / (BOLTZMANN_K * T)
        escape_rate = (
            _c(4.0) * PI * R_exo ** 2 * n_exo * m_co2 * v_th
            * tf.math.exp(-lam)
        )
        dP = -escape_rate * self.gravity / (
            _c(4.0) * PI * self.radius ** 2
        ) * dt
        s.surface_pressure = tf.maximum(s.surface_pressure + dP, _c(0.0))

        # --- Step 4: Ice budget (first-order sublimation) ---
        L_sub = _c(5.7e5)
        A_cap = _c(0.05) * _c(4.0) * PI * self.radius ** 2
        sublimation_flux = STEFAN_BOLTZMANN * T ** 4
        dMice = tf.where(
            s.ice_mass > _c(0.0),
            -A_cap * sublimation_flux / L_sub * dt,
            _c(0.0),
        )
        s.ice_mass = tf.maximum(s.ice_mass + dMice, _c(0.0))
