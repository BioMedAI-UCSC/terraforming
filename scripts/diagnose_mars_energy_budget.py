#!/usr/bin/env python3
"""Diagnose area-weighted Mars radiative and surface-energy fluxes at a restart."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import forcing_with_surface_properties
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.restart import load_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import (
    _co2_surface_tendencies,
    _true_anomaly,
    mean_anomaly_for_ls,
    regolith_conduction_tendencies,
    surface_energy_tendencies,
    two_stream_radiative_fluxes,
)
from run_gcm3d_ablation import ABLATIONS, _ames_dust_climatology


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("ames_reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truncation", default="T21")
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    coords = coordinate_system(args.truncation, args.layers)
    grid = coords.horizontal
    specs = physics_specs(MARS_BODY_3D)
    state = load_restart(args.restart)
    forcing = radiative_forcing(
        diurnal=False,
        co2_radiation_enabled=True,
        dust_visible_optical_depth=0.3,
        dust_longwave_optical_depth=0.1,
    )
    forcing = dataclasses.replace(
        forcing,
        init_orbital_angle_rad=mean_anomaly_for_ls(0.0, forcing),
    )
    forcing = forcing_with_surface_properties(
        forcing, grid, args.surface_properties
    )
    dust_ls, visible, longwave = _ames_dust_climatology(
        args.ames_reference, grid
    )
    forcing = dataclasses.replace(
        forcing,
        **ABLATIONS["convection"],
        dust_climatology_ls_deg=jnp.asarray(dust_ls),
        dust_visible_climatology=jnp.asarray(visible),
        dust_longwave_climatology=jnp.asarray(longwave),
    )
    cf = co2_forcing(energy_limited=True)
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64).copy()
    weights /= weights.sum()
    time_scale_s = 1.0 / float(
        specs.nondimensionalize(1.0 * scales.units.second)
    )
    elapsed_seconds = float(state.dynamics.sim_time) * time_scale_s

    def mean(value) -> float:
        return float(np.sum(np.asarray(value, dtype=np.float64) * weights))

    def diagnose(active_forcing) -> dict[str, float]:
        dyn = state.dynamics
        temperature_nodal = grid.to_nodal(dyn.temperature_variation)
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            grid, dyn.vorticity, dyn.divergence
        )
        pressure_scale = float(
            specs.dimensionalize(1.0, scales.units.pascal).magnitude
        )
        ps_pa = (
            jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0]
            * pressure_scale
        )
        radiation = two_stream_radiative_fluxes(
            state, coords, specs, MARS_BODY_3D, active_forcing,
            surface_pressure_pa=ps_pa,
        )
        _, _, surface = surface_energy_tendencies(
            state,
            coords,
            specs,
            MARS_BODY_3D,
            active_forcing,
            wind_nodal=wind_nodal,
            temperature_nodal=temperature_nodal,
            ps_pa=ps_pa,
        )
        ground_surface, _ = regolith_conduction_tendencies(
            state, specs, active_forcing
        )
        conduction_into_surface = (
            ground_surface[0] / time_scale_s * active_forcing.thermal_inertia
        )
        nonlatent_residual = (
            surface.net_external_w_m2
            - surface.sensible_heat_w_m2
            + conduction_into_surface
        )
        _, _, latent_tendency = _co2_surface_tendencies(
            state,
            coords,
            specs,
            MARS_BODY_3D,
            cf,
            available_surface_flux_w_m2=nonlatent_residual,
            surface_pressure_pa=ps_pa,
        )
        latent_into_surface = (
            latent_tendency[0] / time_scale_s * active_forcing.thermal_inertia
        )
        return {
            "solar_down_toa_w_m2": mean(radiation.shortwave_down_w_m2[0]),
            "solar_up_toa_w_m2": mean(radiation.shortwave_up_w_m2[0]),
            "solar_down_surface_w_m2": mean(radiation.shortwave_down_w_m2[-1]),
            "solar_up_surface_w_m2": mean(radiation.shortwave_up_w_m2[-1]),
            "longwave_up_toa_w_m2": mean(radiation.longwave_up_w_m2[0]),
            "longwave_down_surface_w_m2": mean(radiation.longwave_down_w_m2[-1]),
            "longwave_up_surface_w_m2": mean(radiation.longwave_up_w_m2[-1]),
            "toa_net_down_w_m2": mean(radiation.toa_net_down_w_m2),
            "atmospheric_radiative_convergence_w_m2": mean(
                np.sum(np.asarray(radiation.atmospheric_convergence_w_m2), axis=0)
            ),
            "surface_radiative_net_w_m2": mean(surface.net_external_w_m2),
            "surface_sensible_up_w_m2": mean(surface.sensible_heat_w_m2),
            "conduction_into_surface_w_m2": mean(conduction_into_surface),
            "co2_latent_into_surface_w_m2": mean(latent_into_surface),
            "surface_total_net_w_m2": mean(nonlatent_residual + latent_into_surface),
        }

    corrected = diagnose(forcing)
    legacy = diagnose(dataclasses.replace(forcing, solar_slant_path_enabled=False))
    clear_sky = diagnose(dataclasses.replace(
        forcing,
        dust_visible_optical_depth=0.0,
        dust_longwave_optical_depth=0.0,
        dust_visible_climatology=jnp.zeros_like(forcing.dust_visible_climatology),
        dust_longwave_climatology=jnp.zeros_like(forcing.dust_longwave_climatology),
    ))
    half_co2_lw = diagnose(dataclasses.replace(
        forcing, ames_co2_longwave_opacity_scale=0.5
    ))
    half_all_lw = diagnose(dataclasses.replace(
        forcing,
        ames_co2_longwave_opacity_scale=0.5,
        ames_dust_longwave_opacity_scale=0.5,
    ))
    quarter_all_lw = diagnose(dataclasses.replace(
        forcing,
        ames_co2_longwave_opacity_scale=0.25,
        ames_dust_longwave_opacity_scale=0.25,
    ))
    report = {
        "restart": str(args.restart),
        "restart_sha256": hashlib.sha256(args.restart.read_bytes()).hexdigest(),
        "elapsed_sols": elapsed_seconds / MARS_BODY_3D.rotation_period_s,
        "solar_longitude_deg": float(np.degrees(
            float(_true_anomaly(elapsed_seconds, forcing))
            + forcing.ls_perihelion_rad
        ) % 360.0),
        "corrected_slant_path": corrected,
        "corrected_clear_sky": clear_sky,
        "half_co2_longwave_opacity": half_co2_lw,
        "half_all_longwave_opacity": half_all_lw,
        "quarter_all_longwave_opacity": quarter_all_lw,
        "legacy_vertical_path": legacy,
        "corrected_minus_legacy": {
            key: corrected[key] - legacy[key] for key in corrected
        },
        "interpretation": (
            "Instantaneous area-weighted fluxes at one seasonal state; annual "
            "equilibrium requires integration over a complete Mars year."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
