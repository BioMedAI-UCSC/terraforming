"""Minimal Mars planet state model with a documented evolve() interface.

This module provides a small implementation scaffold that matches the
`Planet` schema documented in `docs/implementation/01_baseline_mars_system.md`.
It is intentionally lightweight: physics terms are simplified and safe defaults
are used when optional fields are not provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict


SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
SECONDS_PER_SOL = 88775.244
PROTON_MASS_KG = 1.67262192369e-27
G_NEWTON = 6.67430e-11
DEFAULT_SPECIES = (
    "CO2",
    "N2",
    "Ar",
    "CO",
    "O",
    "O2",
    "O3",
    "H",
    "H2",
    "He",
    "super_ghg",
    "Ne",
    "Kr",
    "Xe",
)
EARTHLIKE_SOIL_PRESET = {
    "preset_name": "earthlike_loam",
    "chemistry_mass_fraction": {
        "SiO2": 0.50,
        "Al2O3": 0.15,
        "Fe_oxides": 0.08,
        "CaO": 0.05,
        "MgO": 0.04,
        "K2O": 0.03,
        "Na2O": 0.03,
        "organic_matter": 0.08,
        "other": 0.04,
    },
    "water_holding_capacity": 0.25,
    "porosity": 0.45,
    "permeability_m2": 1.0e-12,
    "bulk_density_kg_m3": 1300.0,
    "ph": 6.8,
    "depth_m": 1.0,
}


def _ensure_group(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    group = root.get(key)
    if not isinstance(group, dict):
        group = {}
        root[key] = group
    return group


def _wrap_degrees(value: float) -> float:
    return value % 360.0


def _normalize_vmr_from_columns(columns: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in columns.values())
    if total <= 0.0:
        return {k: 0.0 for k in columns}
    return {k: max(0.0, float(v)) / total for k, v in columns.items()}


@dataclass
class PlanetState:
    """Container for grouped planet data."""

    meta: Dict[str, Any] = field(default_factory=dict)
    static: Dict[str, Any] = field(default_factory=dict)
    dynamic: Dict[str, Any] = field(default_factory=dict)


class PlanetModel:
    """Minimal model interface: initialize, validate_state, diagnostics, evolve."""

    def initialize(
        self, static_data: Dict[str, Any], dynamic_initial: Dict[str, Any], meta: Dict[str, Any] | None = None
    ) -> PlanetState:
        state = PlanetState(meta=meta or {}, static=static_data, dynamic=dynamic_initial)
        self.validate_state(state)
        return state

    def apply_earthlike_soil(
        self, planet: PlanetState, blend: float = 1.0, overrides: Dict[str, Any] | None = None
    ) -> PlanetState:
        """Apply an Earth-like soil preset to static soil/regolith properties.

        Args:
            planet: Current planet state.
            blend: 0..1 blend factor with existing values (1 = full overwrite to preset).
            overrides: Optional dict to tweak any preset fields.
        """
        blend = max(0.0, min(1.0, float(blend)))
        soil = _ensure_group(planet.static, "soil_regolith")
        preset = {**EARTHLIKE_SOIL_PRESET}
        if overrides:
            preset.update(overrides)

        for key, preset_value in preset.items():
            current = soil.get(key)
            if isinstance(preset_value, dict):
                current_dict = current if isinstance(current, dict) else {}
                merged: Dict[str, float] = {}
                keys = set(current_dict.keys()) | set(preset_value.keys())
                for subkey in keys:
                    cv = float(current_dict.get(subkey, 0.0))
                    pv = float(preset_value.get(subkey, 0.0))
                    merged[subkey] = (1.0 - blend) * cv + blend * pv
                total = sum(max(0.0, v) for v in merged.values())
                if total > 0.0:
                    merged = {k: max(0.0, v) / total for k, v in merged.items()}
                soil[key] = merged
            elif isinstance(preset_value, (int, float)):
                cv = float(current) if isinstance(current, (int, float)) else float(preset_value)
                soil[key] = (1.0 - blend) * cv + blend * float(preset_value)
            else:
                soil[key] = preset_value
        return planet

    def validate_state(self, planet: PlanetState) -> None:
        """Check basic structure and non-negative physical quantities."""
        identity = planet.static.get("identity_orbit", {})
        required = ("gravity_m_s2", "mean_radius_m")
        missing = [key for key in required if key not in identity]
        if missing:
            raise ValueError(f"Missing required static.identity_orbit fields: {missing}")

        atmosphere = planet.dynamic.get("atmosphere", {})
        p = atmosphere.get("pressure_Pa")
        rho = atmosphere.get("density_kg_m3")
        t = atmosphere.get("temperature_K")
        if p is not None and p < 0:
            raise ValueError("dynamic.atmosphere.pressure_Pa must be >= 0")
        if rho is not None and rho < 0:
            raise ValueError("dynamic.atmosphere.density_kg_m3 must be >= 0")
        if t is not None and t <= 0:
            raise ValueError("dynamic.atmosphere.temperature_K must be > 0")

    def diagnostics(self, planet: PlanetState) -> Dict[str, float]:
        """Return lightweight derived diagnostics for logging."""
        time_group = planet.dynamic.get("time", {})
        rad = planet.dynamic.get("radiation_energy", {})
        hydro = planet.dynamic.get("hydro_cryosphere", {})
        atmosphere = planet.dynamic.get("atmosphere", {})

        return {
            "sol_elapsed": float(time_group.get("sol_elapsed", 0.0)),
            "solar_longitude_deg": float(time_group.get("solar_longitude_deg", 0.0)),
            "toa_incident_solar_flux_W_m2": float(rad.get("toa_incident_solar_flux_W_m2", 0.0)),
            "surface_incident_solar_flux_W_m2": float(rad.get("surface_incident_solar_flux_W_m2", 0.0)),
            "surface_thermal_ir_flux_W_m2": float(rad.get("surface_thermal_ir_flux_W_m2", 0.0)),
            "pressure_Pa": float(atmosphere.get("pressure_Pa", 0.0)),
            "polar_ice_h2o_kg_m2": float(hydro.get("polar_ice_h2o_kg_m2", 0.0)),
            "polar_ice_co2_kg_m2": float(hydro.get("polar_ice_co2_kg_m2", 0.0)),
        }

    def evolve(
        self,
        planet: PlanetState,
        dt_s: float,
        forcings: Dict[str, Any] | None = None,
        controls: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> PlanetState:
        """Advance dynamic state by one timestep.

        Notes:
        - This is a reduced-order skeleton; plug in higher-fidelity physics later.
        - Static fields are preserved; dynamic groups are updated in place.
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        forcings = forcings or {}
        controls = controls or {}
        params = params or {}
        solar_radiation = forcings.get("solar_radiation", {})
        if not isinstance(solar_radiation, dict):
            solar_radiation = {}
        solar_wind = forcings.get("solar_wind", {})
        if not isinstance(solar_wind, dict):
            solar_wind = {}
        cosmic_radiation = forcings.get("cosmic_radiation", {})
        if not isinstance(cosmic_radiation, dict):
            cosmic_radiation = {}
        giant_planets_gravity = forcings.get("giant_planets_gravity", {})
        if not isinstance(giant_planets_gravity, dict):
            giant_planets_gravity = {}
        mars_moons_gravity = forcings.get("mars_moons_gravity", {})
        if not isinstance(mars_moons_gravity, dict):
            mars_moons_gravity = {}

        dynamic = planet.dynamic
        identity = planet.static.get("identity_orbit", {})
        soil = _ensure_group(planet.static, "soil_regolith")

        time_group = _ensure_group(dynamic, "time")
        atmosphere = _ensure_group(dynamic, "atmosphere")
        composition = _ensure_group(dynamic, "composition")
        plasma = _ensure_group(dynamic, "plasma_magnetic")
        radiation = _ensure_group(dynamic, "radiation_energy")
        hydro = _ensure_group(dynamic, "hydro_cryosphere")
        weather = _ensure_group(dynamic, "weather")

        # 1) Update orbital/solar geometry
        sol_elapsed = float(time_group.get("sol_elapsed", 0.0)) + dt_s / SECONDS_PER_SOL
        time_group["sol_elapsed"] = sol_elapsed

        # Mean-motion approximation for solar longitude.
        n_deg_per_sol = 360.0 / 668.6
        ls0 = float(time_group.get("solar_longitude_deg", 0.0))
        time_group["solar_longitude_deg"] = _wrap_degrees(ls0 + n_deg_per_sol * dt_s / SECONDS_PER_SOL)

        # Mars-Sun distance (AU) from simple Keplerian approximation.
        ecc = float(identity.get("orbital_eccentricity", 0.0934))
        a_au = float(identity.get("semi_major_axis_AU", 1.523679))
        ls_rad = math.radians(time_group["solar_longitude_deg"])
        r_au_default = a_au * (1.0 - ecc**2) / (1.0 + ecc * math.cos(ls_rad))
        r_au = float(solar_radiation.get("sun_mars_distance_AU", r_au_default))

        # Optional gravity influence from giant planets (Jupiter/Saturn) as external forcing.
        jupiter_accel = float(giant_planets_gravity.get("jupiter_accel_m_s2", 0.0))
        saturn_accel = float(giant_planets_gravity.get("saturn_accel_m_s2", 0.0))
        net_giant_accel = jupiter_accel + saturn_accel
        orbit_coupling = float(params.get("giant_gravity_orbit_coupling_au_per_s_per_m_s2", 0.0))
        if orbit_coupling != 0.0:
            r_au += net_giant_accel * orbit_coupling * dt_s
            r_au = max(0.1, r_au)
        time_group["sun_mars_distance_AU"] = r_au
        time_group["sun_mars_distance_m"] = r_au * 149_597_870_700.0
        time_group["jupiter_accel_m_s2"] = jupiter_accel
        time_group["saturn_accel_m_s2"] = saturn_accel
        time_group["net_giant_planets_accel_m_s2"] = net_giant_accel

        # Optional Mars moons gravity diagnostics (Phobos/Deimos) at surface.
        mars_radius_m = float(identity.get("mean_radius_m", 3_389_500.0))
        phobos_mass = float(mars_moons_gravity.get("phobos_mass_kg", 1.0659e16))
        phobos_distance = float(mars_moons_gravity.get("phobos_distance_m", 9_376_000.0))
        deimos_mass = float(mars_moons_gravity.get("deimos_mass_kg", 1.4762e15))
        deimos_distance = float(mars_moons_gravity.get("deimos_distance_m", 23_463_000.0))

        phobos_accel = mars_moons_gravity.get("phobos_accel_m_s2")
        if phobos_accel is None and phobos_distance > 0.0:
            phobos_accel = G_NEWTON * phobos_mass / (phobos_distance * phobos_distance)
        phobos_accel = float(phobos_accel or 0.0)

        deimos_accel = mars_moons_gravity.get("deimos_accel_m_s2")
        if deimos_accel is None and deimos_distance > 0.0:
            deimos_accel = G_NEWTON * deimos_mass / (deimos_distance * deimos_distance)
        deimos_accel = float(deimos_accel or 0.0)

        phobos_tidal = mars_moons_gravity.get("phobos_tidal_accel_m_s2")
        if phobos_tidal is None and phobos_distance > 0.0:
            phobos_tidal = 2.0 * G_NEWTON * phobos_mass * mars_radius_m / (phobos_distance**3)
        phobos_tidal = float(phobos_tidal or 0.0)

        deimos_tidal = mars_moons_gravity.get("deimos_tidal_accel_m_s2")
        if deimos_tidal is None and deimos_distance > 0.0:
            deimos_tidal = 2.0 * G_NEWTON * deimos_mass * mars_radius_m / (deimos_distance**3)
        deimos_tidal = float(deimos_tidal or 0.0)

        net_moons_accel = phobos_accel + deimos_accel
        net_moons_tidal = phobos_tidal + deimos_tidal
        time_group["phobos_accel_m_s2"] = phobos_accel
        time_group["deimos_accel_m_s2"] = deimos_accel
        time_group["phobos_tidal_accel_m_s2"] = phobos_tidal
        time_group["deimos_tidal_accel_m_s2"] = deimos_tidal
        time_group["net_moons_accel_m_s2"] = net_moons_accel
        time_group["net_moons_tidal_accel_m_s2"] = net_moons_tidal

        # 2) Compute top-of-atmosphere forcing
        zenith = float(
            solar_radiation.get(
                "solar_zenith_angle_deg",
                time_group.get("solar_zenith_angle_deg", forcings.get("solar_zenith_angle_deg", 60.0)),
            )
        )
        time_group["solar_zenith_angle_deg"] = zenith
        cosz = max(0.0, math.cos(math.radians(zenith)))

        solar_constant_1au = float(solar_radiation.get("solar_constant_1au_W_m2", params.get("solar_constant_1au_W_m2", 1361.0)))
        toa_flux_model = solar_constant_1au / (r_au * r_au) * cosz
        toa_flux = float(solar_radiation.get("toa_incident_solar_flux_W_m2", toa_flux_model))
        radiation["toa_incident_solar_flux_W_m2"] = toa_flux

        # 3) Integrate atmosphere + composition tendencies (reduced-order)
        transmittance = float(solar_radiation.get("atmospheric_transmittance", params.get("atmospheric_transmittance", 0.55)))
        surface_incident = toa_flux * max(0.0, min(1.0, transmittance))
        radiation["surface_incident_solar_flux_W_m2"] = surface_incident

        albedo = float(soil.get("gcm_surface_bare_ground_albedo", 0.25))
        radiation["surface_reflected_solar_flux_W_m2"] = surface_incident * albedo

        temp_k = float(atmosphere.get("temperature_K", 210.0))
        ir_emissivity_surface = float(params.get("surface_emissivity", 0.95))
        ir_emissivity_down = float(params.get("down_ir_effective_emissivity", 0.2))
        ir_up = ir_emissivity_surface * SIGMA * temp_k**4
        ir_down = ir_emissivity_down * SIGMA * temp_k**4
        radiation["thermal_ir_up_W_m2"] = ir_up
        radiation["thermal_ir_down_W_m2"] = ir_down
        radiation["surface_thermal_ir_flux_W_m2"] = ir_down

        # Cosmic radiation (diagnostic baseline): GCR/SEP flux and dose rates.
        gcr_flux = float(cosmic_radiation.get("gcr_flux_particles_cm2_s", 0.0))
        sep_flux = float(cosmic_radiation.get("sep_flux_particles_cm2_s", 0.0))
        dose_toa = float(cosmic_radiation.get("dose_rate_toa_mSv_day", 0.0))
        shielding = float(cosmic_radiation.get("atmospheric_shielding_factor", 0.0))
        shielding = max(0.0, min(1.0, shielding))
        dose_surface = dose_toa * (1.0 - shielding)
        radiation["cosmic_gcr_flux_particles_cm2_s"] = gcr_flux
        radiation["cosmic_sep_flux_particles_cm2_s"] = sep_flux
        radiation["cosmic_dose_rate_toa_mSv_day"] = dose_toa
        radiation["cosmic_dose_rate_surface_mSv_day"] = dose_surface
        radiation["cosmic_atmospheric_shielding_factor"] = shielding

        # Very simple relaxation on temperature with optional external forcing.
        net_flux = surface_incident - radiation["surface_reflected_solar_flux_W_m2"] + ir_down - ir_up
        dose_to_forcing = float(params.get("cosmic_forcing_W_m2_per_mSv_day", 0.0))
        cosmic_energy_forcing = dose_surface * dose_to_forcing
        extra_forcing = float(
            solar_radiation.get("surface_energy_forcing_W_m2", forcings.get("surface_energy_forcing_W_m2", 0.0))
        )
        heat_capacity = float(params.get("effective_heat_capacity_J_m2_K", 2.0e6))
        atmosphere["temperature_K"] = max(
            1.0, temp_k + (net_flux + extra_forcing + cosmic_energy_forcing) * dt_s / heat_capacity
        )

        # Well-mixed pressure tendency with optional cosmic-radiation escape coupling.
        dp_dt = float(controls.get("pressure_tendency_Pa_s", 0.0))
        cosmic_escape_pressure_coeff = float(params.get("cosmic_escape_pressure_loss_pa_s_per_mSv_day", 0.0))
        cosmic_escape_dp_dt = -dose_surface * max(0.0, cosmic_escape_pressure_coeff)
        giant_gravity_pressure_coupling = float(params.get("giant_gravity_pressure_pa_s_per_m_s2", 0.0))
        giant_gravity_dp_dt = net_giant_accel * giant_gravity_pressure_coupling
        moons_gravity_pressure_coupling = float(params.get("moons_gravity_pressure_pa_s_per_m_s2", 0.0))
        moons_tidal_pressure_coupling = float(params.get("moons_tidal_pressure_pa_s_per_m_s2", 0.0))
        moons_gravity_dp_dt = (
            net_moons_accel * moons_gravity_pressure_coupling + net_moons_tidal * moons_tidal_pressure_coupling
        )
        atmosphere["pressure_Pa"] = max(
            0.0,
            float(atmosphere.get("pressure_Pa", 610.0))
            + (dp_dt + cosmic_escape_dp_dt + giant_gravity_dp_dt + moons_gravity_dp_dt) * dt_s,
        )

        # 4) Plasma/magnetic coupling with external solar-wind inputs
        electron_density = float(
            solar_wind.get("electron_number_density_cm3", forcings.get("electron_number_density_cm3", 0.0))
        )
        magnetic_field_forced = float(solar_wind.get("magnetic_field_nT", forcings.get("magnetic_field_nT", 0.0)))
        wind_speed = float(solar_wind.get("wind_speed_km_s", 0.0))
        proton_density = float(solar_wind.get("proton_density_cm3", 0.0))

        # Dynamic pressure estimate if not explicitly provided (proton-only approximation).
        dynamic_pressure_npa = solar_wind.get("dynamic_pressure_nPa")
        if dynamic_pressure_npa is None and wind_speed > 0.0 and proton_density > 0.0:
            n_m3 = proton_density * 1.0e6
            v_ms = wind_speed * 1.0e3
            dynamic_pressure_pa = n_m3 * PROTON_MASS_KG * v_ms * v_ms
            dynamic_pressure_npa = dynamic_pressure_pa * 1.0e9
        if dynamic_pressure_npa is None:
            dynamic_pressure_npa = 0.0

        # Magnetic field update terms:
        # - optional direct/relaxed forcing to an external target
        # - optional wind-pressure coupling
        # - optional recovery toward a background crustal field
        # - optional manual control tendency
        current_b = float(plasma.get("magnetic_field_nT", magnetic_field_forced))
        manual_db_dt = float(controls.get("magnetic_field_tendency_nT_s", 0.0))
        pressure_coupling = float(params.get("magnetic_pressure_coupling_nT_per_nPa_s", 0.0))
        db_wind_dt = pressure_coupling * float(dynamic_pressure_npa)

        magnetic_forcing_present = ("magnetic_field_nT" in solar_wind) or ("magnetic_field_nT" in forcings)
        forcing_tau = float(params.get("magnetic_forcing_relaxation_timescale_s", 0.0))
        if magnetic_forcing_present and forcing_tau > 0.0:
            db_forcing_dt = (magnetic_field_forced - current_b) / forcing_tau
        elif magnetic_forcing_present:
            db_forcing_dt = (magnetic_field_forced - current_b) / max(dt_s, 1.0)
        else:
            db_forcing_dt = 0.0

        magnetic_static = planet.static.get("magnetic_field", {})
        background_b = float(magnetic_static.get("background_field_nT", current_b))
        recovery_tau = float(params.get("magnetic_background_recovery_timescale_s", 0.0))
        db_recovery_dt = (background_b - current_b) / recovery_tau if recovery_tau > 0.0 else 0.0

        total_db_dt = manual_db_dt + db_wind_dt + db_forcing_dt + db_recovery_dt
        next_b = current_b + total_db_dt * dt_s

        plasma["electron_number_density_cm3"] = electron_density
        plasma["magnetic_field_nT"] = next_b
        plasma["solar_wind_speed_km_s"] = wind_speed
        plasma["solar_wind_proton_density_cm3"] = proton_density
        plasma["solar_wind_dynamic_pressure_nPa"] = float(dynamic_pressure_npa)
        plasma["magnetic_field_tendency_nT_s"] = total_db_dt

        # 5) Integrate hydro/cryosphere tendencies
        hydro["polar_ice_h2o_kg_m2"] = max(
            0.0,
            float(hydro.get("polar_ice_h2o_kg_m2", 0.0))
            + float(controls.get("polar_ice_h2o_tendency_kg_m2_s", 0.0)) * dt_s,
        )
        hydro["polar_ice_co2_kg_m2"] = max(
            0.0,
            float(hydro.get("polar_ice_co2_kg_m2", 0.0))
            + float(controls.get("polar_ice_co2_tendency_kg_m2_s", 0.0)) * dt_s,
        )

        # Bulk water reservoirs (solid/liquid) for non-polar and aggregate hydrology controls.
        hydro["water_ice_column_kg_m2"] = max(
            0.0,
            float(hydro.get("water_ice_column_kg_m2", 0.0))
            + float(controls.get("water_ice_tendency_kg_m2_s", 0.0)) * dt_s,
        )
        hydro["water_liquid_column_kg_m2"] = max(
            0.0,
            float(hydro.get("water_liquid_column_kg_m2", 0.0))
            + float(controls.get("water_liquid_tendency_kg_m2_s", 0.0)) * dt_s,
        )

        # Optional phase-change exchange term (positive means melt: ice -> liquid).
        phase_change = float(controls.get("water_phase_change_tendency_kg_m2_s", 0.0))
        transfer = phase_change * dt_s
        if transfer > 0.0:
            actual = min(transfer, hydro["water_ice_column_kg_m2"])
            hydro["water_ice_column_kg_m2"] -= actual
            hydro["water_liquid_column_kg_m2"] += actual
        elif transfer < 0.0:
            actual = min(-transfer, hydro["water_liquid_column_kg_m2"])
            hydro["water_liquid_column_kg_m2"] -= actual
            hydro["water_ice_column_kg_m2"] += actual

        # 5b) Integrate atmospheric species columns and VMRs.
        columns = composition.get("column_kg_m2")
        if not isinstance(columns, dict):
            columns = {}
        for species in DEFAULT_SPECIES:
            columns.setdefault(species, 0.0)

        # Add any user-provided extra species without restricting names.
        tendency_map = controls.get("species_column_tendency_kg_m2_s", {})
        if not isinstance(tendency_map, dict):
            tendency_map = {}

        # Cosmic-radiation-driven composition/escape tendencies.
        ozone_loss_coeff = float(params.get("cosmic_ozone_loss_kg_m2_s_per_mSv_day", 0.0))
        h_loss_coeff = float(params.get("cosmic_h_escape_kg_m2_s_per_mSv_day", 0.0))
        h2_loss_coeff = float(params.get("cosmic_h2_escape_kg_m2_s_per_mSv_day", 0.0))
        if ozone_loss_coeff > 0.0:
            tendency_map["O3"] = float(tendency_map.get("O3", 0.0)) - dose_surface * ozone_loss_coeff
        if h_loss_coeff > 0.0:
            tendency_map["H"] = float(tendency_map.get("H", 0.0)) - dose_surface * h_loss_coeff
        if h2_loss_coeff > 0.0:
            tendency_map["H2"] = float(tendency_map.get("H2", 0.0)) - dose_surface * h2_loss_coeff
        for species in tendency_map:
            columns.setdefault(species, 0.0)

        for species, tendency in tendency_map.items():
            columns[species] = max(0.0, float(columns.get(species, 0.0)) + float(tendency) * dt_s)
        composition["column_kg_m2"] = columns

        vmr = composition.get("vmr")
        if not isinstance(vmr, dict):
            vmr = {}
        vmr.update(_normalize_vmr_from_columns(columns))

        # Optional direct VMR controls (for scenario forcing), renormalized to sum=1.
        vmr_target = controls.get("species_vmr_target", {})
        if isinstance(vmr_target, dict) and vmr_target:
            for species, value in vmr_target.items():
                vmr[species] = max(0.0, float(value))
            vmr_sum = sum(vmr.values())
            if vmr_sum > 0.0:
                vmr = {k: v / vmr_sum for k, v in vmr.items()}
        composition["vmr"] = vmr
        composition["cosmic_escape_tendency_kg_m2_s"] = {
            "O3": -dose_surface * max(0.0, ozone_loss_coeff),
            "H": -dose_surface * max(0.0, h_loss_coeff),
            "H2": -dose_surface * max(0.0, h2_loss_coeff),
        }
        atmosphere["cosmic_escape_pressure_tendency_Pa_s"] = cosmic_escape_dp_dt
        atmosphere["giant_gravity_pressure_tendency_Pa_s"] = giant_gravity_dp_dt
        atmosphere["moons_gravity_pressure_tendency_Pa_s"] = moons_gravity_dp_dt

        # 5c) Regolith compound changes (static soil chemistry updates).
        chemistry = soil.get("chemistry_mass_fraction")
        if not isinstance(chemistry, dict):
            chemistry = {}

        # Time-derivative style control: d(fraction)/dt per compound.
        soil_tendency = controls.get("soil_compound_tendency_mass_fraction_per_s", {})
        if not isinstance(soil_tendency, dict):
            soil_tendency = {}
        for compound in soil_tendency:
            chemistry.setdefault(compound, 0.0)
        for compound, tendency in soil_tendency.items():
            chemistry[compound] = max(0.0, float(chemistry.get(compound, 0.0)) + float(tendency) * dt_s)

        # Direct per-step delta style control.
        soil_delta = controls.get("soil_compound_delta_mass_fraction", {})
        if not isinstance(soil_delta, dict):
            soil_delta = {}
        for compound in soil_delta:
            chemistry.setdefault(compound, 0.0)
        for compound, delta in soil_delta.items():
            chemistry[compound] = max(0.0, float(chemistry.get(compound, 0.0)) + float(delta))

        # Keep normalized mass fractions when any chemistry is present.
        chem_total = sum(max(0.0, float(v)) for v in chemistry.values())
        if chem_total > 0.0:
            chemistry = {k: max(0.0, float(v)) / chem_total for k, v in chemistry.items()}
        soil["chemistry_mass_fraction"] = chemistry

        # 6) Weather diagnostics and constraint checks
        weather["season_label"] = _season_label(time_group["solar_longitude_deg"])
        composition.setdefault("vmr", {})
        composition.setdefault("column_kg_m2", {})

        self.validate_state(planet)
        return planet


def _season_label(ls_deg: float) -> str:
    ls = _wrap_degrees(ls_deg)
    if ls < 90.0:
        return "northern_spring"
    if ls < 180.0:
        return "northern_summer"
    if ls < 270.0:
        return "northern_autumn"
    return "northern_winter"


if __name__ == "__main__":
    model = PlanetModel()
    state = model.initialize(
        static_data={
            "identity_orbit": {
                "gravity_m_s2": 3.71,
                "mean_radius_m": 3_389_500.0,
                "semi_major_axis_AU": 1.523679,
                "orbital_eccentricity": 0.0934,
            },
            "soil_regolith": {"gcm_surface_bare_ground_albedo": 0.25},
        },
        dynamic_initial={
            "time": {"solar_longitude_deg": 0.0, "solar_zenith_angle_deg": 60.0},
            "atmosphere": {"temperature_K": 210.0, "pressure_Pa": 610.0, "density_kg_m3": 0.02},
            "hydro_cryosphere": {"polar_ice_h2o_kg_m2": 0.0, "polar_ice_co2_kg_m2": 0.0},
        },
    )
    state = model.evolve(state, dt_s=3600.0)
    print(model.diagnostics(state))
