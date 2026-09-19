"""Mars-specific configuration for the reusable framework GCM physics."""

from __future__ import annotations

from src.celestials.planets.mars import planet as mars
from src.framework.physics.gcm import CO2Forcing, RadiativeForcing


def radiative_forcing(
    albedo: float = 0.25,
    greenhouse_factor: float = 1.02,
    diurnal: bool = True,
    init_orbital_angle_rad: float = 0.0,
    co2_radiation_enabled: bool = False,
    dust_visible_optical_depth: object = 0.0,
    dust_longwave_optical_depth: object = 0.0,
) -> RadiativeForcing:
    """Build GCM radiative forcing from the canonical Mars constants."""
    return RadiativeForcing(
        albedo=albedo,
        greenhouse_factor=greenhouse_factor,
        emissivity=float(mars.MARS_SURFACE_EMISSIVITY),
        stefan_boltzmann=float(mars.STEFAN_BOLTZMANN),
        thermal_inertia=float(mars.MARS_THERMAL_INERTIA),
        rotation_period_s=float(mars.MARS_ROTATION_PERIOD),
        axial_tilt_rad=float(mars.MARS_AXIAL_TILT),
        ls_perihelion_rad=float(mars.MARS_LS_PERIHELION),
        orbital_period_s=float(mars.MARS_ORBITAL_PERIOD),
        semi_major_axis_m=float(mars.MARS_SEMI_MAJOR_AXIS),
        eccentricity=float(mars.MARS_ECCENTRICITY),
        init_orbital_angle_rad=init_orbital_angle_rad,
        diurnal=diurnal,
        co2_radiation_enabled=co2_radiation_enabled,
        dust_visible_optical_depth=dust_visible_optical_depth,
        dust_longwave_optical_depth=dust_longwave_optical_depth,
    )


def co2_forcing(
    exchange_rate_pa_s_per_k: float = 1.0e-4,
    escape_rate_kg_s: float = 0.0,
    energy_limited: bool = True,
) -> CO2Forcing:
    """Build CO2-cycle forcing from the canonical Mars constants."""
    return CO2Forcing(
        frost_point_k=float(mars.MARS_CO2_FROST_POINT),
        latent_heat_j_kg=float(mars.MARS_CO2_LATENT_HEAT),
        gravity_m_s2=float(mars.MARS_BODY_3D.gravity_m_s2),
        thermal_inertia=float(mars.MARS_THERMAL_INERTIA),
        exchange_rate_pa_s_per_k=exchange_rate_pa_s_per_k,
        escape_rate_kg_s=escape_rate_kg_s,
        energy_limited=energy_limited,
    )


__all__ = [
    "co2_forcing",
    "radiative_forcing",
]
