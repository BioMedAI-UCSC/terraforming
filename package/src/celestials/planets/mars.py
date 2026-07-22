"""Mars – concrete implementation of the abstract Planet (PyTorch backend).

Mars provides the **physics model** — the governing equations for how its
atmosphere, surface temperature, and ice budget evolve.  It does **not**
own the integration strategy; that belongs to the engine (TimeController).

Physics supplied to the engine:
    ``compute_derivatives(y)``   – coupled ODE RHS [dT/dt, dP/dt, dM_ice/dt]
    ``compute_fast_physics(dt)`` – reduced-order analytic/relaxation update

Physical constants are from the NASA Mars Fact Sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html

GPU / device support
--------------------
Pass ``device='cuda'`` (or any valid torch device string) to run all state
tensors and physics computations on that device.  Module-level constants are
CPU tensors; they are moved to the planet's device during ``setup_properties``
and cached as ``self._*`` instance attributes so every subsequent kernel call
stays on-device with no cross-device copies.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from src.constants import (
    TF_DTYPE,
    STEFAN_BOLTZMANN,
)

from src.framework.planet import Planet
from src.framework.atmosphere import Atmosphere
from src.framework.thermal import Thermal
from src.framework.water import Water
from src.framework.radiation import Radiation
from src.framework.magnetic import Magnetic
from src.framework.intrinsic import IntrinsicParameters
from src.framework.orbital import OrbitalParameters

# ---------------------------------------------------------------------------
# Mars-specific constants  (module-level CPU tensors; moved to device in
# setup_properties via self._* caching — never used directly in hot paths)
# ---------------------------------------------------------------------------
MARS_MASS: torch.Tensor             = torch.tensor(6.4171e23,   dtype=TF_DTYPE)  # kg
MARS_RADIUS: torch.Tensor           = torch.tensor(3.3895e6,    dtype=TF_DTYPE)  # m
MARS_GRAVITY: torch.Tensor          = torch.tensor(3.72076,     dtype=TF_DTYPE)  # m s⁻²
MARS_ROTATION_PERIOD: torch.Tensor  = torch.tensor(88_775.244,  dtype=TF_DTYPE)  # s  (1 sol)
MARS_SEMI_MAJOR_AXIS: torch.Tensor  = torch.tensor(2.27939200e11, dtype=TF_DTYPE) # m  (1.524 AU)
MARS_ECCENTRICITY: torch.Tensor     = torch.tensor(0.0934,      dtype=TF_DTYPE)  # dimensionless
MARS_ORBITAL_PERIOD: torch.Tensor   = torch.tensor(5.93568e7,   dtype=TF_DTYPE)  # s  (~687 d)
MARS_AXIAL_TILT: torch.Tensor        = torch.tensor(25.19 * math.pi / 180.0, dtype=TF_DTYPE)  # rad

# Physics constants calibrated to Mars observations
MARS_LS_PERIHELION: torch.Tensor      = torch.tensor(251.0 * math.pi / 180.0, dtype=TF_DTYPE)  # rad
MARS_SURFACE_EMISSIVITY: torch.Tensor = torch.tensor(0.95,     dtype=TF_DTYPE)
MARS_THERMAL_INERTIA: torch.Tensor    = torch.tensor(6.0e4,    dtype=TF_DTYPE)  # J K⁻¹ m⁻²
MARS_MAVEN_ESCAPE_RATE: torch.Tensor  = torch.tensor(0.2,      dtype=TF_DTYPE)  # kg s⁻¹
MARS_CO2_FROST_POINT: torch.Tensor    = torch.tensor(149.0,    dtype=TF_DTYPE)  # K
MARS_CO2_LATENT_HEAT: torch.Tensor    = torch.tensor(5.7e5,    dtype=TF_DTYPE)  # J kg⁻¹
# Effective fractional surface area of each sublimating/condensing seasonal CO2
# cap, per pole. This sets the amplitude of the seasonal CO2-cycle pressure swing
# in compute_derivatives / compute_fast_physics (dMice -> dP). The former 0.01
# was far too small: it produced only a ~16% peak-to-peak swing, versus the
# ~25-30% seen by the Viking Landers (Hess et al. 1980; Tillman et al. 1993) and
# reproduced by MCD 6.1. Calibrated to that observed swing: 0.023 yields ~25%,
# within the observed band. Still conservative — the real seasonal cap reaches
# mid-latitudes — this is an *effective* area folding in partial coverage and
# sublimation efficiency. See docs/validation/baseline.md.
MARS_POLAR_CAP_FRACTION: torch.Tensor = torch.tensor(0.023,    dtype=TF_DTYPE)  # dimensionless
MARS_DIURNAL_SWING_AMP: torch.Tensor  = torch.tensor(50.0,     dtype=TF_DTYPE)  # K
MARS_THERMAL_TIDE_PA: torch.Tensor    = torch.tensor(30.0,     dtype=TF_DTYPE)  # Pa
MARS_THERMAL_TIDE_PHASE: torch.Tensor = torch.tensor(-0.7 * math.pi, dtype=TF_DTYPE)  # rad

# Default atmospheric composition (partial pressures in Pa)
MARS_DEFAULT_COMPOSITION: Dict[str, torch.Tensor] = {
    "CO2": torch.tensor(580.0, dtype=TF_DTYPE),
    "N2":  torch.tensor(15.0,  dtype=TF_DTYPE),
    "Ar":  torch.tensor(12.0,  dtype=TF_DTYPE),
    "O2":  torch.tensor(0.8,   dtype=TF_DTYPE),
    "CO":  torch.tensor(0.4,   dtype=TF_DTYPE),
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
    smooth_gates : bool, optional
        If True, replace the hard ice-exhaustion gate (sublimation snaps to
        zero when a cap empties) with a smooth tanh fade so gradients
        stay nonzero through cap depletion.  Default False — the hard gate
        keeps existing runs bit-identical.
    ice_ref_kg : float, optional
        Reference ice mass (kg) setting the width of the smooth gate:
        gate = tanh(M_ice / ice_ref_kg).  Only used when ``smooth_gates=True``.
        Default 10¹² kg.
    device : str or torch.device, optional
        PyTorch device for all state tensors.  Default ``'cpu'``.
        Pass ``'cuda'`` (or ``'cuda:0'``) to run on GPU.
    """

    def __init__(
        self,
        surface_temperature: float = 210.0,
        surface_pressure: float = 610.0,
        albedo: float = 0.25,
        # Mars's CO₂ atmosphere is very thin (610 Pa vs Earth's 101,325 Pa — 166× thinner).
        greenhouse_factor: float = 1.02,
        composition: Optional[Dict[str, float]] = None,
        ice_mass: float = 5.0e15,
        latitude: float = 22.0,
        longitude: float = 0.0,
        elevation_m: float = 0.0,
        initial_ls_deg: float = 251.0,
        smooth_gates: bool = False,
        ice_ref_kg: float = 1.0e12,
        device: str | torch.device | None = None,
    ) -> None:
        if ice_ref_kg <= 0.0:
            raise ValueError(f"ice_ref_kg must be positive, got {ice_ref_kg}")
        self._smooth_gates = bool(smooth_gates)
        self._init_ice_ref = float(ice_ref_kg)
        # Store device first — setup_properties() reads it.
        # Default: CUDA if available, otherwise CPU.
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        d = self._device

        def _t(v: float) -> torch.Tensor:
            """Create a scalar tensor on the planet's device."""
            return torch.tensor(v, dtype=TF_DTYPE, device=d)

        # All IntrinsicParameters and OrbitalParameters tensors must be on device
        # so that distance_from_sun() and physics methods stay device-local.
        self.intrinsic_params = IntrinsicParameters(
            mass=MARS_MASS.to(d),
            radius=MARS_RADIUS.to(d),
            gravity=MARS_GRAVITY.to(d),
            rotation_period=MARS_ROTATION_PERIOD.to(d),
        )

        self.orbital_params = OrbitalParameters(
            semi_major_axis=MARS_SEMI_MAJOR_AXIS.to(d),
            eccentricity=MARS_ECCENTRICITY.to(d),
            orbital_period=MARS_ORBITAL_PERIOD.to(d),
            axial_tilt=MARS_AXIAL_TILT.to(d),
        )

        # Hydrostatic elevation correction: P = P_ref * exp(-z / H)
        # Mars CO₂ scale height H ≈ 11.1 km at mean surface temperature.
        MARS_SCALE_HEIGHT_M = 11_100.0
        corrected_pressure = surface_pressure * math.exp(-elevation_m / MARS_SCALE_HEIGHT_M)

        # Initial-condition tensors — all on device
        self._init_temperature    = _t(surface_temperature)
        self._init_pressure       = _t(corrected_pressure)
        self._init_albedo         = _t(albedo)
        self._init_greenhouse     = _t(greenhouse_factor)
        self._init_ice_mass       = _t(ice_mass)
        self._init_latitude       = _t(latitude * math.pi / 180.0)
        self._init_longitude      = _t(longitude * math.pi / 180.0)
        # orbital_angle = 0 is perihelion; Ls = orbital_angle + Ls_perihelion
        self._init_orbital_angle  = _t((initial_ls_deg - 251.0) * math.pi / 180.0)

        if composition is not None:
            self._init_composition = {k: _t(v) for k, v in composition.items()}
        else:
            self._init_composition = {k: v.to(d) for k, v in MARS_DEFAULT_COMPOSITION.items()}

        # Composition is the single source of truth for what the atmosphere
        # contains: rescale the partial pressures so they sum exactly to the
        # (elevation-corrected) total surface pressure.
        comp_total = sum(self._init_composition.values())
        self._init_composition = {
            k: v * (self._init_pressure / comp_total)
            for k, v in self._init_composition.items()
        }

        self.setup_properties()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    def setup_properties(self) -> None:
        """Initialize the planet's physical properties and cache device-local
        copies of all module-level constants used in the physics hot-paths."""
        d = self._device

        # ---- Planetary state dataclasses (all tensors on device) ----
        _ls_deg = float(self._init_orbital_angle.item()) * 180.0 / math.pi + 251.0
        _ls_deg = _ls_deg % 360.0
        if 0.0 <= _ls_deg < 180.0:
            _f_north = 0.0
        else:
            _f_north = 0.4 * (_ls_deg - 180.0) / 180.0
        _f_south = 1.0 - _f_north

        # Atmospheric mass follows hydrostatically from the surface pressure:
        # M_atm = P · 4πR² / g  (the old hardcoded 2.5e16 kg was ~6 % high).
        _A_planet = 4.0 * math.pi * self.intrinsic_params.radius ** 2
        self.atmosphere = Atmosphere(
            surface_pressure=self._init_pressure.clone(),
            atmospheric_mass=(
                self._init_pressure * _A_planet / self.intrinsic_params.gravity
            ),
            composition=dict(self._init_composition),
        )
        self.thermal = Thermal(
            surface_temperature=self._init_temperature.clone(),
            greenhouse_factor=self._init_greenhouse.clone(),
        )
        self.water = Water(
            ice_mass=self._init_ice_mass.clone(),
            ice_mass_north=self._init_ice_mass * torch.tensor(_f_north, dtype=TF_DTYPE, device=d),
            ice_mass_south=self._init_ice_mass * torch.tensor(_f_south, dtype=TF_DTYPE, device=d),
            liquid_mass=torch.zeros((), dtype=TF_DTYPE, device=d),
            vapour_mass=torch.tensor(1.0e13, dtype=TF_DTYPE, device=d),
        )
        self.radiation = Radiation(
            albedo=self._init_albedo.clone(),
            solar_flux=torch.zeros((), dtype=TF_DTYPE, device=d),
        )
        self.magnetic = Magnetic(
            magnetic_field_strength=torch.tensor(5.0e-9, dtype=TF_DTYPE, device=d),
        )
        self.elapsed_time  = torch.zeros((), dtype=TF_DTYPE, device=d)
        self.orbital_angle = self._init_orbital_angle.clone()

        # ---- GHG intervention baselines (populated on first inject) ----
        self._baseline_ghf = None          # torch.Tensor | None — CO₂-only GHF at t=0
        self._baseline_olr = None          # torch.Tensor | None — ε σ (T₀/GHF₀)⁴

        # ---- Device-local constant cache ----
        # These are the only references used inside compute_derivatives and
        # compute_fast_physics so that no CPU tensor ever touches GPU math.
        self._SB          = STEFAN_BOLTZMANN.to(d)                                    # σ
        self._TI          = MARS_THERMAL_INERTIA.to(d)                                # C (J K⁻¹ m⁻²)
        self._EMISS       = MARS_SURFACE_EMISSIVITY.to(d)                             # ε
        self._LS_PERI     = MARS_LS_PERIHELION.to(d)                                  # Ls at perihelion
        self._CAP_FRAC    = MARS_POLAR_CAP_FRACTION.to(d)
        self._Q_out_pole  = (MARS_SURFACE_EMISSIVITY * STEFAN_BOLTZMANN              # ε σ T_frost⁴
                             * MARS_CO2_FROST_POINT ** 4).to(d)
        self._LAT_HEAT    = MARS_CO2_LATENT_HEAT.to(d)
        self._ESCAPE_RATE = MARS_MAVEN_ESCAPE_RATE.to(d)
        self._TIDE_PA     = MARS_THERMAL_TIDE_PA.to(d)
        self._TIDE_PHASE  = MARS_THERMAL_TIDE_PHASE.to(d)
        self._DIURNAL_AMP = MARS_DIURNAL_SWING_AMP.to(d)
        self._ICE_REF     = torch.tensor(self._init_ice_ref, dtype=TF_DTYPE, device=d)

    # ==================================================================
    # PHYSICS: shared gates
    # ==================================================================
    def _gate_sublimation(self, dM: torch.Tensor, ice: torch.Tensor) -> torch.Tensor:
        """Stop the sublimation flux (dM < 0) of an exhausted polar cap.

        Hard mode (default): dM snaps to zero when the cap is empty.
        Correct physics, but its gradient w.r.t. everything is zero once a
        cap empties — the CO₂-collapse / cap-depletion regime is invisible
        to gradient-based optimisation.

            gate:  dM ← 0        where  M_ice ≤ 0  and  dM < 0

        Smooth mode (``smooth_gates=True``): the sublimating branch is
        multiplied by tanh(M_ice / ice_ref).  While M_ice ≫ ice_ref the gate
        saturates to exactly 1.0 in float64, so the physics is unchanged;
        as M_ice → 0 the flux fades continuously to exactly 0 instead of
        snapping off, while d(flux)/d(M_ice) = dM / ice_ref stays nonzero at
        the empty cap — the optimizer can see through cap exhaustion.

            gate:  dM ← dM · tanh(M_ice / ice_ref)    where  dM < 0

        tanh is used rather than the sigmoid of the original spec because
        σ(0) = ½ would keep sublimating mass out of an *empty* cap forever
        (a mass-conservation leak); tanh(0) = 0 removes the leak and matches
        the hard gate exactly at both extremes.  Both integration paths
        clamp ice ≥ 0, so the gate argument is never negative.

        Condensation (dM > 0) is never gated in either mode.
        """
        if self._smooth_gates:
            return torch.where(
                dM < 0.0,
                dM * torch.tanh(ice / self._ICE_REF),
                dM,
            )
        return torch.where(
            (ice <= 0.0) & (dM < 0.0),
            torch.zeros_like(dM),
            dM,
        )

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

        All arithmetic uses self._* cached constants so no CPU tensor is
        introduced when the planet runs on CUDA.

        References
        ----------
        Stefan-Boltzmann law : https://en.wikipedia.org/wiki/Stefan–Boltzmann_law
        Jeans escape         : https://en.wikipedia.org/wiki/Atmospheric_escape
        """
        s = self

        T     = y[0].clamp(min=1.0)
        P     = y[1].clamp(min=0.0)
        M_ice = y[2].clamp(min=0.0)  # noqa: F841  (available for future use)

        # ==================================================================
        # --- dT/dt: energy balance ---
        # ==================================================================
        # We compute for a specific 1 km^2 patch (per unit area formulation).
        # Area A cancels out (1 m² effective) for local patch heating/cooling.

        # Hour angle h: t=0 is local midnight → h = -π
        omega = 2.0 * math.pi / self.intrinsic_params.rotation_period
        h     = omega * s.elapsed_time - math.pi

        lat = self._init_latitude

        # True anomaly is s.orbital_angle (0 = perihelion).
        Ls    = s.orbital_angle + self._LS_PERI
        delta = torch.asin(torch.sin(self.orbital_params.axial_tilt) * torch.sin(Ls))

        # Diurnal incidence angle (Lambert's cosine law)
        cos_zenith = (
            torch.sin(lat) * torch.sin(delta)
            + torch.cos(lat) * torch.cos(delta) * torch.cos(h)
        ).clamp(min=0.0)

        Q_in  = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith
        T_eff = T / s.thermal.greenhouse_factor.clamp(min=1.0)
        Q_out = self._EMISS * self._SB * T_eff ** 4

        dT_dt = (Q_in - Q_out) / self._TI

        # ==================================================================
        # --- dM_ice/dt: Polar CO₂ Sublimation & Condensation ---
        # ==================================================================
        A_cap = self._CAP_FRAC * 4.0 * math.pi * self.intrinsic_params.radius ** 2

        cos_zenith_N = torch.sin(delta).clamp(min=0.0)
        cos_zenith_S = (-torch.sin(delta)).clamp(min=0.0)

        Q_in_N = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_N
        Q_in_S = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * cos_zenith_S

        net_sub_N = (Q_in_N - self._Q_out_pole) * A_cap / self._LAT_HEAT
        net_sub_S = (Q_in_S - self._Q_out_pole) * A_cap / self._LAT_HEAT

        # Gate sublimation (dMice < 0) when that pole's ice is exhausted.
        dMice_N = self._gate_sublimation(-net_sub_N, s.water.ice_mass_north)
        dMice_S = self._gate_sublimation(-net_sub_S, s.water.ice_mass_south)
        dMice_dt = dMice_N + dMice_S

        # ==================================================================
        # --- dP/dt: Non-thermal escape + ice cap mass exchange ---
        # ==================================================================
        # CO₂ Jeans (thermal) escape is negligible for Mars (λ≈299).
        # Non-thermal loss parameterised by MAVEN data (Jakosky et al. 2018).
        # P is the global-mean (mass-budget) pressure: every term is an
        # explicit mass flux, so atmosphere + caps + escape·t is conserved.
        # The local thermal tide is a diagnostic overlay
        # (observed_surface_pressure), not a mass source.
        A_planet = 4.0 * math.pi * self.intrinsic_params.radius ** 2
        dP_dt = (
            -self._ESCAPE_RATE * self.intrinsic_params.gravity / A_planet
            + (-dMice_dt * self.intrinsic_params.gravity / A_planet)
        )

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

        3. Update ice mass via first-order sublimation.

        4. Update pressure via a first-order escape term.

        References
        ----------
        Energy balance : https://scied.ucar.edu/learning-zone/how-climate-works/energy-balance
        Relaxation     : "Newtonian cooling" approximation
        """
        s = self

        # --- Step 1: Equilibrium temperature (latitude-aware) ---
        Ls    = s.orbital_angle + self._LS_PERI
        delta = torch.asin(torch.sin(self.orbital_params.axial_tilt) * torch.sin(Ls))
        lat   = self._init_latitude

        cos_h0 = (-torch.tan(lat) * torch.tan(delta)).clamp(-1.0, 1.0)
        h0     = torch.acos(cos_h0)
        insolation_factor = (
            h0 * torch.sin(lat) * torch.sin(delta)
            + torch.cos(lat) * torch.cos(delta) * torch.sin(h0)
        ) / math.pi
        insolation_factor = insolation_factor.clamp(min=0.0)

        absorbed   = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * insolation_factor
        T_eq_base  = (absorbed / (self._EMISS * self._SB)).clamp(min=0.0) ** 0.25
        T_eq       = T_eq_base * s.thermal.greenhouse_factor

        # Superimpose diurnal variation: minimum at midnight, maximum at noon.
        omega = 2.0 * math.pi / self.intrinsic_params.rotation_period
        swing = self._DIURNAL_AMP * torch.cos(lat)
        T_eq  = T_eq - swing * torch.cos(omega * s.elapsed_time)

        # --- Step 2: Exponential relaxation ---
        T_cur = s.thermal.surface_temperature.clamp(min=1.0)
        tau   = (self._TI / (4.0 * self._EMISS * self._SB * T_cur ** 3)).clamp(min=1.0)
        s.thermal.surface_temperature = (
            T_eq + (T_cur - T_eq) * torch.exp(-dt / tau)
        ).clamp(min=1.0)

        # --- Step 3: Ice budget (Polar CO₂ Sublimation & Condensation) ---
        A_cap = self._CAP_FRAC * 4.0 * math.pi * self.intrinsic_params.radius ** 2

        cos_N = torch.sin(delta).clamp(min=0.0)
        cos_S = (-torch.sin(delta)).clamp(min=0.0)

        Q_N = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * cos_N
        Q_S = (1.0 - s.radiation.albedo) * s.radiation.solar_flux * cos_S

        net_sub_N = (Q_N - self._Q_out_pole) * A_cap / self._LAT_HEAT
        net_sub_S = (Q_S - self._Q_out_pole) * A_cap / self._LAT_HEAT

        dMice_N = self._gate_sublimation(-net_sub_N * dt, s.water.ice_mass_north)
        dMice_S = self._gate_sublimation(-net_sub_S * dt, s.water.ice_mass_south)
        dMice = dMice_N + dMice_S

        # clamp(min=0) has zero gradient at the floor — acceptable: the smooth
        # gate fades the flux before the floor is reached (audit item D2).
        s.water.ice_mass_north = (s.water.ice_mass_north + dMice_N).clamp(min=0.0)
        s.water.ice_mass_south = (s.water.ice_mass_south + dMice_S).clamp(min=0.0)
        s.water.ice_mass       = s.water.ice_mass_north + s.water.ice_mass_south

        # --- Step 4: Pressure (non-thermal escape + ice mass exchange) ---
        # Global-mean mass-budget pressure only — the local thermal tide is
        # applied as a diagnostic overlay in observed_surface_pressure().
        A_planet = 4.0 * math.pi * self.intrinsic_params.radius ** 2
        s.atmosphere.surface_pressure = (
            s.atmosphere.surface_pressure
            - self._ESCAPE_RATE * self.intrinsic_params.gravity / A_planet * dt
            - dMice * self.intrinsic_params.gravity / A_planet
        ).clamp(min=0.0)
        self._sync_composition_co2()

    # ==================================================================
    # PRESSURE BOOKKEEPING — composition is the single source of truth
    # ==================================================================

    def _sync_composition_co2(self) -> None:
        """Keep ``composition["CO2"]`` consistent with the evolving total.

        All bulk pressure change (polar-cap exchange, atmospheric escape) is
        CO₂ — the other species neither condense nor escape at these rates —
        so the CO₂ partial pressure is the total minus the inert partials:

            P_CO2 = P_total − Σ P_i (i ≠ CO2)

        The floor at 0 covers deep-collapse states where the total approaches
        the inert background; it is unreachable in current-Mars regimes.
        """
        others = sum(
            v for k, v in self.atmosphere.composition.items() if k != "CO2"
        )
        self.atmosphere.composition["CO2"] = (
            self.atmosphere.surface_pressure - others
        ).clamp(min=0.0)

    def unpack_state(self, y: torch.Tensor) -> None:
        """Unpack [T, P, M_ice] and re-sync composition to the new pressure."""
        super().unpack_state(y)
        self._sync_composition_co2()

    def observed_surface_pressure(self) -> torch.Tensor:
        """Local (observed) surface pressure: mass-budget mean + thermal tide.

        The diurnal thermal tide redistributes mass around the planet; it is
        a *local* pressure oscillation, not a global source or sink, so it is
        excluded from the prognostic (mass-budget) ``surface_pressure`` and
        applied here as a closed-form overlay:

            P_obs(t) = P_mean(t) + A_tide · [cos(ωt + φ) − cos(φ)]

        This is exactly the time integral of the tide term the integrators
        used to accumulate (−A_tide·ω·sin(ωt + φ)), so observed pressure is
        analytically identical to the old trajectory — now exact instead of
        discretised, and the mass budget closes without it.
        """
        omega = 2.0 * math.pi / self.intrinsic_params.rotation_period
        tide = self._TIDE_PA * (
            torch.cos(omega * self.elapsed_time + self._TIDE_PHASE)
            - torch.cos(self._TIDE_PHASE)
        )
        return self.atmosphere.surface_pressure + tide

    # ==================================================================
    # GHG INTERVENTION — state lives in atmosphere.composition
    # ==================================================================

    def _init_ghg(self) -> None:
        """Cache CO₂-only baseline GHF and OLR at the moment injection begins."""
        from src.interventions.forcing import _MARS_EMISSIVITY
        self._baseline_ghf = self.thermal.greenhouse_factor.clone()
        sb = STEFAN_BOLTZMANN.to(self._device)
        T0 = self.thermal.surface_temperature.clone()
        self._baseline_olr = (
            _MARS_EMISSIVITY * sb * (T0 / self._baseline_ghf) ** 4.0
        )

    def inject(self, schedule: dict[str, float | torch.Tensor]) -> None:
        """Add GHGs to atmosphere.composition (kg → Pa) and resync GHF.

        Each compound's partial pressure increases by ΔP = M·g / A_surface.
        Unknown compounds are added to composition on first injection.
        After this call ``mars.thermal.greenhouse_factor`` and ``mars.delta_F``
        reflect the updated atmospheric state.

        Masses may be tensors: ΔP is computed in tensor arithmetic so
        gradients flow from an injected mass to the greenhouse factor.
        """
        if self._baseline_ghf is None:
            self._init_ghg()
        g = self.intrinsic_params.gravity
        A = 4.0 * math.pi * self.intrinsic_params.radius ** 2
        dP_total = torch.zeros((), dtype=TF_DTYPE, device=self._device)
        for name, kg in schedule.items():
            dP = torch.as_tensor(kg, dtype=TF_DTYPE, device=self._device) * g / A
            if name in self.atmosphere.composition:
                self.atmosphere.composition[name] = self.atmosphere.composition[name] + dP
            else:
                self.atmosphere.composition[name] = dP
            dP_total = dP_total + dP
        # Injected mass raises the total pressure — the mass budget must see
        # what the forcing sees.
        self.atmosphere.surface_pressure = self.atmosphere.surface_pressure + dP_total
        self._recompute_greenhouse_factor()

    def decay_ghg(self, dt_years: float) -> None:
        """Exponentially decay all COMPOUNDS-registered gases in composition.

        Only species present in the COMPOUNDS registry (the seven super-GHGs)
        are decayed. Background gases (CO₂, N₂, Ar) are unaffected.
        """
        from src.interventions.compounds import COMPOUNDS, get_compound
        dP_lost = torch.zeros((), dtype=TF_DTYPE, device=self._device)
        for name in list(self.atmosphere.composition.keys()):
            if name in COMPOUNDS:
                tau = get_compound(name).atmospheric_lifetime_yr
                decay_factor = math.exp(-dt_years / tau)
                before = self.atmosphere.composition[name]
                self.atmosphere.composition[name] = before * decay_factor
                dP_lost = dP_lost + before * (1.0 - decay_factor)
        # Decayed mass leaves the atmosphere — remove it from the total too.
        self.atmosphere.surface_pressure = (
            self.atmosphere.surface_pressure - dP_lost
        ).clamp(min=0.0)
        self._recompute_greenhouse_factor()

    @property
    def delta_F(self) -> torch.Tensor:
        """Total GHG radiative forcing in W/m² from atmosphere composition."""
        from src.interventions.forcing import delta_F_from_composition
        return delta_F_from_composition(
            self.atmosphere.composition,
            self.atmosphere.surface_pressure,
        )

    def _recompute_greenhouse_factor(self) -> None:
        """Sync ``thermal.greenhouse_factor`` from current atmosphere composition.

            GHF_new = GHF_base × (1 + relu(ΔF) / F_in_base)^(1/4)

        relu reproduces the "no forcing when ΔF ≤ 0" semantics without a
        data-dependent branch, so the computation stays on-device and the
        autograd graph from ΔF to GHF is never cut.
        """
        if self._baseline_ghf is None or self._baseline_olr is None:
            return
        dF = self.delta_F
        GHF_new = self._baseline_ghf * torch.pow(
            1.0 + torch.relu(dF) / self._baseline_olr, 0.25
        )
        self.thermal.greenhouse_factor = GHF_new.clamp(min=1.0)
