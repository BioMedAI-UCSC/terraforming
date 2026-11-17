#!/usr/bin/env python3
"""Run restartable Mars-year soil spin-up and incremental PBL ablations.

The default is intentionally a climate experiment (668 sols), not the short map
demo. Use ``--sols`` for smoke tests. Each chunk writes a versioned restart; a
stopped run resumes with ``--resume`` without reinitialising the soil column.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

# Configure XLA before importing NumPy/JAX or any gcm3d module. The defaults are
# intentionally laptop-friendly; expert users can override XLA_FLAGS explicitly.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
)
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    str(Path("outputs/.jax_compilation_cache").resolve()),
)
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import numpy as np
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic

from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.benchmarks import dry_conserved_quantities
from src.framework.gcm.specs import physics_specs
from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.maps import (
    forcing_with_surface_properties,
    initial_rest_state,
    plot_maps,
    run_maps,
    save_netcdf,
)
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.topography import regrid_to_nodal
from src.framework.gcm.restart import load_restart, save_restart
from src.framework.physics.gcm import (
    _true_anomaly,
    dust_optical_depths,
    initial_column_state,
    mean_anomaly_for_ls,
)


ABLATIONS = {
    "drag": dict(regolith_enabled=False, stability_exchange_enabled=False,
                 pbl_diffusion_enabled=False, convective_adjustment_enabled=False),
    "regolith": dict(regolith_enabled=True, stability_exchange_enabled=False,
                     pbl_diffusion_enabled=False, convective_adjustment_enabled=False),
    "stability": dict(regolith_enabled=True, stability_exchange_enabled=True,
                      pbl_diffusion_enabled=False, convective_adjustment_enabled=False),
    "pbl": dict(regolith_enabled=True, stability_exchange_enabled=True,
                pbl_diffusion_enabled=True, convective_adjustment_enabled=False),
    "convection": dict(regolith_enabled=True, stability_exchange_enabled=True,
                       pbl_diffusion_enabled=True, convective_adjustment_enabled=True),
}


def _checkpoint_diagnostics(state, coords, specs, dt_seconds: float, body=MARS_BODY_3D) -> dict[str, float | int]:
    """Return paper-facing global diagnostics from a checkpointed column state."""
    grid = coords.horizontal
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    normalized_weights = weights / np.sum(weights)
    pressure_scale = float(specs.dimensionalize(1.0, scales.units.pascal).magnitude)
    temperature_scale = float(specs.dimensionalize(1.0, scales.units.kelvin).magnitude)
    velocity_scale = float(
        specs.dimensionalize(1.0, scales.units.meter / scales.units.second).magnitude
    )

    surface_pressure_pa = (
        np.exp(np.asarray(grid.to_nodal(state.dynamics.log_surface_pressure))[0])
        * pressure_scale
    )
    co2_ice_pa = np.asarray(state.co2_ice)[0] * pressure_scale
    surface_temperature_k = np.asarray(state.surface_temperature)[0] * temperature_scale
    ground_temperature_k = np.asarray(state.ground_temperature) * temperature_scale
    reference_k = np.full(
        (coords.vertical.layers, 1, 1), body.reference_temperature_k
    )
    air_temperature_k = (
        np.asarray(grid.to_nodal(state.dynamics.temperature_variation))
        * temperature_scale
        + reference_k
    )
    u_nd, v_nd = spherical_harmonic.vor_div_to_uv_nodal(
        grid, state.dynamics.vorticity, state.dynamics.divergence
    )
    wind_speed_ms = np.hypot(np.asarray(u_nd), np.asarray(v_nd)) * velocity_scale
    dry = dry_conserved_quantities(
        state.dynamics,
        coords,
        specs,
        body,
        np.full(coords.vertical.layers, body.reference_temperature_k),
    )
    elapsed_seconds = float(state.dynamics.sim_time) / float(
        specs.nondimensionalize(1.0 * scales.units.second)
    )
    step = round(elapsed_seconds / dt_seconds)
    mean_pressure_pa = float(np.sum(surface_pressure_pa * normalized_weights))
    mean_ice_pa = float(np.sum(co2_ice_pa * normalized_weights))
    from src.framework.physics.gcm import cloud_pressure_equivalent
    cloud_pa = np.asarray(cloud_pressure_equivalent(state, coords))[0] * pressure_scale
    mean_cloud_pa = float(np.sum(cloud_pa * normalized_weights))
    co2_mass_kg = (
        (mean_pressure_pa + mean_ice_pa)
        * 4.0 * np.pi * body.radius_m**2
        / body.gravity_m_s2
    )
    return {
        "step": step,
        "elapsed_sols": elapsed_seconds / body.solar_day_seconds,
        "mean_surface_pressure_pa": mean_pressure_pa,
        "mean_co2_ice_pa": mean_ice_pa,
        "mean_co2_cloud_ice_pa": mean_cloud_pa,
        "atmosphere_plus_surface_reservoir_mass_kg": co2_mass_kg,
        "total_co2_mass_kg": co2_mass_kg + mean_cloud_pa * body.surface_area_m2 / body.gravity_m_s2,
        "mean_surface_temperature_k": float(
            np.sum(surface_temperature_k * normalized_weights)
        ),
        "min_surface_temperature_k": float(np.min(surface_temperature_k)),
        "max_surface_temperature_k": float(np.max(surface_temperature_k)),
        "min_air_temperature_k": float(np.min(air_temperature_k)),
        "max_air_temperature_k": float(np.max(air_temperature_k)),
        "max_wind_speed_ms": float(np.max(wind_speed_ms)),
        "mean_deep_soil_temperature_k": float(
            np.sum(ground_temperature_k[-1] * normalized_weights)
        ),
        "dry_atmospheric_mass_kg_m2_sr": dry.mass_kg_m2_sr,
        "atmospheric_energy_excluding_surface_geopotential_j_m2_sr": dry.total_energy_j_m2_sr,
        "dry_atmospheric_aam_kg_m_s_sr": dry.axial_angular_momentum_kg_m_s_sr,
    }


def _append_checkpoint_diagnostics(path: Path, record: dict[str, float | int]) -> None:
    """Append one monotonic checkpoint record, safely tolerating exact retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if reader.fieldnames != list(record):
                raise ValueError(
                    f"diagnostic schema changed for {path}: "
                    f"existing={reader.fieldnames}, new={list(record)}"
                )
        if rows:
            last_step = int(float(rows[-1]["step"]))
            step = int(record["step"])
            if last_step == step:
                return
            if last_step > step:
                raise ValueError(
                    f"diagnostic history {path} ends at step {last_step}, "
                    f"ahead of restart step {step}"
                )
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


@functools.lru_cache(maxsize=2)
def _load_ames_dust(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as opened:
        return opened[["dust_visible_optical_depth", "dust_longwave_optical_depth"]].load()


def _ames_dust_at_ls(path: Path, grid, ls_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate Ames five-sol dust fields periodically to a Dinosaur grid."""
    import xarray as xr

    dust = _load_ames_dust(path)
    seasonal = xr.concat(
        [dust.isel(ls=-1).assign_coords(ls=float(dust.ls[-1]) - 360.0),
         dust,
         dust.isel(ls=0).assign_coords(ls=float(dust.ls[0]) + 360.0)],
        dim="ls",
    ).interp(ls=ls_deg % 360.0)
    periodic = xr.concat(
        [seasonal.isel(lon=-1).assign_coords(lon=float(seasonal.lon[-1]) - 360.0),
         seasonal,
         seasonal.isel(lon=0).assign_coords(lon=float(seasonal.lon[0]) + 360.0)],
        dim="lon",
    )
    target = periodic.interp(
        lon=np.mod(np.degrees(np.asarray(grid.longitudes)), 360.0),
        lat=np.degrees(np.asarray(grid.latitudes)),
    )
    return tuple(
        np.asarray(target[name].transpose("lon", "lat"))
        for name in ("dust_visible_optical_depth", "dust_longwave_optical_depth")
    )


def _ames_dust_climatology(path: Path, grid):
    """Regrid the complete Ames seasonal dust table to Dinosaur nodal points."""
    import xarray as xr

    dust = _load_ames_dust(path)
    periodic = xr.concat(
        [dust.isel(lon=-1).assign_coords(lon=float(dust.lon[-1]) - 360.0),
         dust,
         dust.isel(lon=0).assign_coords(lon=float(dust.lon[0]) + 360.0)],
        dim="lon",
    )
    target = periodic.interp(
        lon=np.mod(np.degrees(np.asarray(grid.longitudes)), 360.0),
        lat=np.degrees(np.asarray(grid.latitudes)),
    )
    return (
        np.asarray(target.ls),
        np.asarray(target.dust_visible_optical_depth.transpose("ls", "lon", "lat")),
        np.asarray(target.dust_longwave_optical_depth.transpose("ls", "lon", "lat")),
    )


def reference_configuration(grid,data_directory,dt_seconds,*,moment_water=True,dust_opacity_from_aerosol=False,
                            scale_surface_flux=False,solar_constant_w_m2=None,linear_k_interpolation=False,
                            shortwave_ramp_sols=-1.,fixed_calendar_date=-1,surface_map_options=None):
    """Released present-Mars cloud switches from build_run/fms_mars_clouds_v3.2."""
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    from src.celestials.planets.mars.surface import reference_surface_fields_on_grid
    from src.celestials.planets.mars.boundary_inputs import polar_water_on_grid
    from src.celestials.planets.mars.dust_scenario import load_scenario_on_grid,load_injection_fractions_on_grid,load_local_time_fractions_on_grid
    from src.framework.physics.soil_frost import subsurface_properties
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER
    data_directory=Path(data_directory)
    fields=reference_surface_fields_on_grid(grid,data_directory,**(surface_map_options or {}))
    diffusivity,capacity=subsurface_properties(jnp.asarray(grid.latitudes)[None,:],
        fields['thermal_inertia'],fields['grs_ice_latitude_deg'])
    scenario=load_scenario_on_grid(grid,data_directory/'DustScenario_MY30.nc',map_scale=3.67)
    forcing=dataclasses.replace(reference_radiative_forcing(solar_constant_w_m2=solar_constant_w_m2),albedo=jnp.asarray(fields['albedo']),
        emissivity=jnp.asarray(fields['emissivity']),co2_frost_radiative_properties=True,
        water_frost_radiative_properties=True,diurnal=True)
    radiation_interval=8*dt_seconds
    radiation=make_radiation_component(make_particle_provider(bands=12,
        dust_tracers=(('dust_mass','dust_number'),),
        water_moment_tracers=(WATER_CLOUD_TRACER,'water_core_mass','water_ice_number') if moment_water else None),
        radiation_interval_seconds=radiation_interval,radiation_mode='12band',scale_surface_flux=scale_surface_flux,
        linear_k_interpolation=linear_k_interpolation,shortwave_ramp_sols=shortwave_ramp_sols,
        fixed_calendar_date=fixed_calendar_date)
    options=dict(soil_diffusivity=diffusivity,soil_volumetric_heat_capacity=capacity,reference_pressure_pa=705.,
        radiation_update_seconds=radiation_interval,moment_water=moment_water,moment_dust=not moment_water,
        dust_opacity_channel=1,dust_optical_bands=12,
        initial_water_surface_kg_m2=polar_water_on_grid(grid,data_directory),
        initial_water_vapor_ratio=0.,initial_dust_mass_ratio=0.,initial_dust_number_per_kg=0.,
        rayleigh_options=dict(model_top=True),
        dust_background_options=dict(scenario=scenario,effective_radius_m=2e-6),
        moment_cloud_options=dict(microtimestep=30.,contact=.95,latent_heat=False) if moment_water else None)
    if dust_opacity_from_aerosol:
        from src.framework.physics.prescribed_dust import make_fixed_dust_provider
        forcing=dataclasses.replace(forcing,dust_visible_optical_depth=0.)
        options['dust_opacity_from_aerosol']=True
        options['dust_background_options'].pop('scenario')
        options['dust_background_options']['fixed_dust_provider']=make_fixed_dust_provider(bands=12,scenario=scenario,
            fixed_calendar_date=fixed_calendar_date)
    fraction_path=data_directory/'radius_scenario.nc'
    if fraction_path.is_file():
        options['dust_background_options']['injection_fraction_scenario']=load_injection_fractions_on_grid(
            grid,fraction_path,1)
    local_time_path=data_directory/'localtime_frac.nc'
    if local_time_path.is_file():
        options['dust_background_options']['local_time_scenario']=load_local_time_fractions_on_grid(grid,local_time_path)
    inputs=('mars_thermal_inertia8_2011.nc','mars_TES_albedo8_new.nc','mars_sfc_emissivity8.nc',
        'GRS_interp16.nc','npcflag8.nc','DustScenario_MY30.nc')
    for optional in (fraction_path,local_time_path):
        if optional.is_file(): inputs+=(optional.name,)
    hashes={name:hashlib.sha256((data_directory/name).read_bytes()).hexdigest() for name in inputs}
    return AMES_BODY_3D,forcing,radiation,options,hashes


def reference_rest_state(coords,specs,body,elevation_nodal_m,*,temperature_k=170.,isothermal=True):
    """Released hydro_eq cold start, with Dinosaur's spectral/sigma geometry."""
    from src.framework.physics.hydrostatic_initialization import surface_pressure,air_temperature
    from src.framework.physics.vertical_grid import coefficients
    from src.framework.gcm.dynamics import reference_temperature
    grid=coords.horizontal
    dynamics=initial_rest_state(coords,specs,body,np.zeros(grid.nodal_shape),t_ref_k=170.,p0_pa=705.)
    scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    top=coefficients(56,705.)[0][0]
    pressure=surface_pressure(body.gravity_m_s2*jnp.asarray(elevation_nodal_m),grid.quadrature_weights,
        temperature_k=temperature_k,gas_constant=body.gas_constant_j_kg_k,top_pressure_pa=top)
    log_pressure=grid.clip_wavenumbers(grid.to_modal(jnp.log(pressure[None]/scale)))
    resolved=jnp.exp(grid.to_nodal(log_pressure))[0]*scale
    half=jnp.asarray(coords.vertical.boundaries)[:,None,None]*resolved
    temperature=air_temperature(half,resolved,temperature_k=temperature_k,isothermal=isothermal,
        gravity=body.gravity_m_s2,cp=body.cp_j_kg_k)
    kelvin=float(specs.nondimensionalize(1.*scales.units.kelvin))
    reference=jnp.asarray(reference_temperature(coords,body))[:,None,None]
    variation=grid.to_modal(temperature*kelvin-reference)
    return dataclasses.replace(dynamics,log_surface_pressure=log_pressure,temperature_variation=variation)


def reference_surface_temperature_k(latitude_rad,*,equilibrated=True,soil_temperature_k=170.):
    """Released equilibrated-latitude or uniform soil cold start."""
    latitude=jnp.asarray(latitude_rad)
    if not equilibrated: return jnp.full_like(latitude,soil_temperature_k)
    return jnp.where(latitude < -15.*jnp.pi/180.,
        165.+55.5*.5*(1.+jnp.cos((latitude+.261799)*180./75.)),
        167.+52.*jnp.cos((latitude+.261799)*90./105.))


def alternate_reference_radiation(coords,mode,profile=None,timescale=-4.,fixed_calendar_date=-1):
    """Select a released alternate branch and return its restart identity."""
    if mode=='simple':
        if profile is not None: raise ValueError('simple radiation does not read an equilibrium profile')
        from src.framework.physics.simple_radiation import make_radiation_component
        metadata=dict(
            radiation_mode='simple',radiation_update_seconds=None,simple_geometry_interval_seconds=1800.)
        if fixed_calendar_date>=0:
            metadata['reference_radiation_controls']=dict(fixed_calendar_date=fixed_calendar_date)
        return make_radiation_component(fixed_calendar_date=fixed_calendar_date),dict(radiation_update_seconds=None),metadata
    if fixed_calendar_date>=0: raise ValueError('fixed radiation date requires spectral or simple radiation')
    if mode not in ('fixed','height') or profile is None:
        raise ValueError('fixed or height relaxation requires an equilibrium profile')
    if not np.isfinite(timescale): raise ValueError('thermal timescale must be finite')
    from src.framework.physics.thermal_relaxation import load_equilibrium
    from src.framework.physics.vertical_grid import pressure_interfaces
    half=np.asarray(pressure_interfaces(coords.vertical.layers,705.,705.))
    values=load_equilibrium(coords.horizontal,profile,half,height=mode=='height')
    thermal=dict(equilibrium=values[0] if mode=='height' else values,timescale=timescale)
    metadata=dict(radiation_mode='thermal-'+mode,radiation_update_seconds=None,
        thermal_timescale=timescale,equilibrium_profile_sha256=hashlib.sha256(Path(profile).read_bytes()).hexdigest())
    if mode=='height':
        thermal.update(source_height=values[1],top_pressure_pa=float(half[0]))
        metadata['diagnostic_top_pressure_pa']=float(half[0])
    return None,dict(radiation_update_seconds=None,thermal_relaxation_options=thermal),metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_properties", type=Path,
        help="surface dataset, or released input-data directory with --config reference")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gcm3d_ablation"))
    parser.add_argument("--config", choices=["all", "reference", *ABLATIONS], default="all")
    parser.add_argument("--mola-path",type=Path,help="explicit staged MOLA raster; required for reference runs")
    parser.add_argument('--reference-radiation',choices=('spectral','simple','fixed','height'),default='spectral')
    parser.add_argument('--scale-surface-ir-flux',action='store_true',
        help='use native do_irflux_scale to normalize surface infrared emission to sigma*T^4')
    parser.add_argument('--solar-constant',type=float,default=None,
        help='solar flux in W/m2 at mean orbital distance; enables spectral scaling; simple default 589')
    parser.add_argument('--linear-k-interpolation',action='store_true',
        help='select native use_boxinterp12=false: linear k and water interpolation with T/log-pressure extrapolation')
    parser.add_argument('--shortwave-ramp-sols',type=float,default=-1.,
        help='native rampup_rad duration in model sols; nonpositive values disable the ramp')
    parser.add_argument('--fixed-radiation-date',type=int,default=-1,
        help='native fixed_calender_date in model days; negative values keep the evolving calendar')
    parser.add_argument('--equilibrium-profile',type=Path,help='released teq/zzf NetCDF input for fixed or height relaxation')
    parser.add_argument('--thermal-timescale',type=float,default=-4.,help='negative values use 86400-second days; positive values use seconds; zero disables')
    parser.add_argument('--moment-water',action=argparse.BooleanOptionalAction,default=True,
        help='disable to run transported dust without water exchange or cloud microphysics')
    parser.add_argument('--freeze-tracer-fields',action='store_true',help='discard atmospheric physics tracer changes after microphysics')
    parser.add_argument('--no-forcing',action='store_true',help='run dynamics without atmospheric or surface physics')
    parser.add_argument('--recalculate-heat-flux',action='store_true',help='use solved PBL temperature for the surface sensible heat flux')
    parser.add_argument('--particle-scavenging',action=argparse.BooleanOptionalAction,default=True,
        help='settle dust cores with water clouds; disable to retain cores in the atmosphere')
    parser.add_argument('--moment-clouds',dest='form_moment_clouds',action=argparse.BooleanOptionalAction,default=True,
        help='form and grow moment water clouds; surface exchange and settling remain independent')
    parser.add_argument('--lifting-suppresses-settling',action='store_true',
        help='retain bottom-layer lifted dust and suppress its surface deposit, as in native lifting_nosed')
    parser.add_argument('--dust-opacity-from-aerosol',action='store_true',
        help='use prescribed aerosol optics as the lifting target without current-opacity pressure scaling')
    parser.add_argument('--gust-factor',type=float,default=0.,help='released dust-lifting wind correction; gust speed is 1 m/s')
    parser.add_argument('--co2-exchange-ratio-pole',type=float,default=.92)
    parser.add_argument('--co2-exchange-ratio-trop',type=float,default=.991)
    for flag,destination in (('co2-condensation','do_co2_condensation'),('atmos-co2-loss','do_atmos_co2_loss'),
                             ('co2-mass-feedback','do_co2_condensation_cycle')):
        parser.add_argument('--'+flag,dest=destination,action=argparse.BooleanOptionalAction,default=True)
    for flag,destination in (('mars-surface','do_mars_surface'),('vertical-diffusion','do_vert_diff'),
                             ('reference-pbl','do_ames_pbl')):
        parser.add_argument('--'+flag,dest=destination,action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--surface-condensation',action=argparse.BooleanOptionalAction,default=True,
        help='enable surface CO2 phase adjustment; disabling requires bare soil in the native solver')
    parser.add_argument('--soil-alpha',type=float,default=.5,help='implicit soil-diffusion weighting (native default 0.5)')
    parser.add_argument('--equilibrated-surface',action=argparse.BooleanOptionalAction,default=True,
        help='use the released latitude profile; disable for uniform initial soil temperature')
    parser.add_argument('--soil-temperature',type=float,default=170.,help='uniform initial soil temperature in K')
    parser.add_argument('--initial-air-temperature',type=float,default=170.,help='native isotemp cold-start temperature in K')
    parser.add_argument('--isothermal-atmosphere',action=argparse.BooleanOptionalAction,default=True,
        help='native do_isothermal cold start; disable for its frost-limited lapse profile')
    parser.add_argument('--north-inertia',type=float,default=-900.,help='replace thermal inertia north of the cap latitude when positive')
    parser.add_argument('--north-inertia-latitude',type=float,default=65.,help='north thermal-inertia boundary in degrees')
    parser.add_argument('--north-inertia-maximum',type=float,default=800.,help='thermal-inertia cap north of the boundary')
    parser.add_argument('--regional-inertia-scale',type=float,default=-100.,help='positive thermal-inertia multiplier strictly between -50 and +30 degrees')
    parser.add_argument("--truncation", default="T21")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--initial-ls", type=float, default=0.0)
    parser.add_argument("--diurnal", action="store_true", help="resolve local-time forcing instead of daily-mean sunlight")
    parser.add_argument("--sols", type=float, default=668.0)
    parser.add_argument("--chunk-sols", type=float, default=10.0)
    parser.add_argument(
        "--hyperdiffusion-tau-sols",
        type=float,
        default=0.1,
        help=(
            "e-folding time in sols for fourth-order diffusion at the highest "
            "resolved wavenumber (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--cooldown-seconds", type=float, default=5.0,
        help="idle time after every checkpoint to limit sustained laptop load",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dust-visible", type=float, default=0.3)
    parser.add_argument("--dust-longwave", type=float, default=0.1)
    parser.add_argument("--co2-lw-scale", type=float, default=1.0)
    parser.add_argument("--dust-lw-scale", type=float, default=1.0)
    parser.add_argument("--surface-exchange-multiplier", type=float, default=1.0)
    parser.add_argument("--ames-dust-reference", type=Path,
                        help="seasonally prescribe visible/IR dust from a staged Ames surface reference")
    parser.add_argument(
        "--laptop", action="store_true",
        help="use T21/12 levels, 300 s steps, 5-sol chunks and 15 s cooldowns",
    )
    args = parser.parse_args()
    reference=args.config=='reference'
    if args.scale_surface_ir_flux and (not reference or args.reference_radiation!='spectral'):
        parser.error('surface infrared scaling requires --config reference and spectral radiation')
    if args.linear_k_interpolation and (not reference or args.reference_radiation!='spectral'):
        parser.error('linear k interpolation requires --config reference and spectral radiation')
    if not np.isfinite(args.shortwave_ramp_sols): parser.error('shortwave ramp duration must be finite')
    if args.shortwave_ramp_sols!=-1. and (not reference or args.reference_radiation!='spectral'):
        parser.error('shortwave ramp requires --config reference and spectral radiation')
    if args.fixed_radiation_date!=-1 and (not reference or args.reference_radiation not in ('spectral','simple')):
        parser.error('fixed radiation date requires --config reference and spectral or simple radiation')
    if args.solar_constant is not None:
        if not reference or args.reference_radiation not in ('spectral','simple'):
            parser.error('solar constant scaling requires --config reference and spectral or simple radiation')
        if not np.isfinite(args.solar_constant) or args.solar_constant<=0.:
            parser.error('solar constant must be finite and positive')
    solar_constant=589. if args.reference_radiation=='simple' and args.solar_constant is None else args.solar_constant
    if not np.isfinite(args.soil_temperature) or args.soil_temperature<=0.:
        parser.error('initial soil temperature must be finite and positive')
    if not np.isfinite(args.initial_air_temperature) or args.initial_air_temperature<=0.:
        parser.error('initial air temperature must be finite and positive')
    if (args.initial_air_temperature!=170. or not args.isothermal_atmosphere) and not reference:
        parser.error('initial atmosphere controls require --config reference')
    if not np.isfinite(args.soil_alpha): parser.error('soil alpha must be finite')
    soil_controls={}
    if not args.surface_condensation: soil_controls['surface_condensation']=False
    if args.soil_alpha!=.5: soil_controls['soil_alpha']=args.soil_alpha
    if soil_controls and not reference: parser.error('soil solver controls require --config reference')
    initial_surface_controls={}
    if not args.equilibrated_surface: initial_surface_controls['equilibrated']=False
    if args.soil_temperature!=170.: initial_surface_controls['soil_temperature_k']=args.soil_temperature
    if initial_surface_controls and (not reference or not args.do_mars_surface):
        parser.error('initial soil controls require --config reference with Mars surface physics')
    surface_map_defaults=dict(north_inertia=-900.,north_latitude_deg=65.,north_maximum=800.,regional_scale=-100.)
    surface_map_options=dict(north_inertia=args.north_inertia,north_latitude_deg=args.north_inertia_latitude,
        north_maximum=args.north_inertia_maximum,regional_scale=args.regional_inertia_scale)
    if (not all(np.isfinite(value) for value in surface_map_options.values()) or
        abs(args.north_inertia_latitude)>90. or args.north_inertia_maximum<=0.):
        parser.error('surface inertia controls require finite values, a valid latitude and a positive cap')
    surface_map_controls={name:value for name,value in surface_map_options.items() if value!=surface_map_defaults[name]}
    if surface_map_controls and not reference: parser.error('surface inertia controls require --config reference')
    process_defaults=dict(freeze_tracer_fields=False,no_forcing=False,gust_factor=0.,recalculate_heat_flux=False,
        particle_scavenging=True,form_moment_clouds=True,lifting_suppresses_settling=False,dust_opacity_from_aerosol=False,
        co2_exchange_ratio_pole=.92,co2_exchange_ratio_trop=.991)
    process_controls={name:getattr(args,name) for name,default in process_defaults.items() if getattr(args,name)!=default}
    for name in ('co2_exchange_ratio_pole','co2_exchange_ratio_trop'):
        value=getattr(args,name)
        if not np.isfinite(value) or value<=0.: parser.error('CO2 exchange ratios must be finite and positive')
    if not np.isfinite(args.gust_factor): parser.error('--gust-factor must be finite')
    if not reference and process_controls: parser.error('reference process controls require --config reference')
    if not reference and not args.moment_water: parser.error('--no-moment-water requires --config reference')
    if not reference and (args.reference_radiation!='spectral' or args.equilibrium_profile is not None or args.thermal_timescale!=-4.):
        parser.error('alternate reference radiation requires --config reference')
    co2_controls={name:getattr(args,name) for name in
        ('do_co2_condensation','do_atmos_co2_loss','do_co2_condensation_cycle')}
    if not reference and not all(co2_controls.values()):
        parser.error('CO2 process switches require --config reference')
    surface_controls={name:getattr(args,name) for name in ('do_mars_surface','do_vert_diff','do_ames_pbl')}
    if not reference and not all(surface_controls.values()):
        parser.error('surface process switches require --config reference')
    if not args.do_mars_surface and args.reference_radiation=='spectral':
        parser.error('the no-surface runner requires simple, fixed or height forcing without transported surface dust/water')
    thermal=args.reference_radiation in ('fixed','height')
    if thermal!=(args.equilibrium_profile is not None):
        parser.error('fixed and height relaxation require --equilibrium-profile; other modes do not use it')
    if args.equilibrium_profile is not None and not args.equilibrium_profile.is_file():
        parser.error('equilibrium profile must be an existing NetCDF file')
    if not np.isfinite(args.thermal_timescale) or (not thermal and args.thermal_timescale!=-4.):
        parser.error('--thermal-timescale must be finite and requires fixed or height relaxation')
    if args.dt is None: args.dt=231. if reference else 300.
    if args.layers is None: args.layers=56 if reference else 12
    if reference and (args.mola_path is None or not args.mola_path.is_file()):
        parser.error('reference runs require --mola-path with an existing staged raster')
    if reference and (args.ames_dust_reference is not None or
        (args.co2_lw_scale,args.dust_lw_scale,args.surface_exchange_multiplier)!=(1.,1.,1.)):
        parser.error('reference physics uses its native radiation and surface controls')
    if args.laptop:
        args.truncation = "T21"
        args.layers = 12
        args.dt = 300.0
        args.chunk_sols = min(args.chunk_sols, 5.0)
        args.cooldown_seconds = max(args.cooldown_seconds, 15.0)
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    for key in ("sols", "chunk_sols", "dt", "hyperdiffusion_tau_sols"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    for key in ("co2_lw_scale", "dust_lw_scale", "surface_exchange_multiplier"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and non-negative")
    if not np.isfinite(args.initial_ls):
        parser.error("--initial-ls must be finite")
    if args.ames_dust_reference is not None and not args.ames_dust_reference.exists():
        parser.error(f"Ames dust reference does not exist: {args.ames_dust_reference}")
    jax.config.update("jax_enable_x64", True)

    verified_dt = {"T42": 450.0, "T85": 300.0, "T106": 225.0, "T170": 150.0}
    if args.truncation in verified_dt and args.dt > verified_dt[args.truncation]:
        parser.error(
            f"--dt={args.dt:g} s exceeds the verified {args.truncation} limit "
            f"of {verified_dt[args.truncation]:g} s"
        )

    configs = list(ABLATIONS) if args.config == "all" else [args.config]
    sigma=None
    if reference:
        from src.framework.physics.vertical_grid import sigma_boundaries as reference_sigma_boundaries
        sigma=reference_sigma_boundaries(args.layers)
    coords = coordinate_system(args.truncation, args.layers,sigma_boundaries=sigma)
    grid = coords.horizontal
    body=MARS_BODY_3D;radiation=None;reference_options=None;input_hashes={};alternate_radiation_metadata={}
    if reference:
        body,base,radiation,reference_options,input_hashes=reference_configuration(
            grid,args.surface_properties,args.dt,moment_water=args.moment_water,
            dust_opacity_from_aerosol=args.dust_opacity_from_aerosol,scale_surface_flux=args.scale_surface_ir_flux,
            solar_constant_w_m2=solar_constant,linear_k_interpolation=args.linear_k_interpolation,
            shortwave_ramp_sols=args.shortwave_ramp_sols,fixed_calendar_date=args.fixed_radiation_date,
            surface_map_options=surface_map_controls)
        reference_options.update(co2_controls)
        reference_options.update(surface_controls)
        reference_options.update(soil_controls)
        reference_options.update(process_controls)
        if not args.do_mars_surface:
            reference_options.update(moment_water=False,moment_dust=False,dust_background_options=None,moment_cloud_options=None)
        input_hashes['mola']=hashlib.sha256(args.mola_path.read_bytes()).hexdigest()
        if args.reference_radiation!='spectral':
            radiation,overrides,alternate_radiation_metadata=alternate_reference_radiation(
                coords,args.reference_radiation,args.equilibrium_profile,args.thermal_timescale,
                fixed_calendar_date=args.fixed_radiation_date)
            reference_options.update(overrides)
            if thermal: input_hashes['equilibrium_profile']=alternate_radiation_metadata['equilibrium_profile_sha256']
    else:
        base = radiative_forcing(
            diurnal=args.diurnal, co2_radiation_enabled=True,
            dust_visible_optical_depth=args.dust_visible,
            dust_longwave_optical_depth=args.dust_longwave,
        )
        base = dataclasses.replace(
            base,
            ames_co2_longwave_opacity_scale=args.co2_lw_scale,
            ames_dust_longwave_opacity_scale=args.dust_lw_scale,
            surface_exchange_multiplier=args.surface_exchange_multiplier,
        )
        base = forcing_with_surface_properties(base, grid, args.surface_properties)
    specs = physics_specs(body)
    base = dataclasses.replace(base, init_orbital_angle_rad=mean_anomaly_for_ls(
        np.deg2rad(args.initial_ls), base
    ))
    if args.ames_dust_reference is not None:
        dust_ls, dust_visible_table, dust_longwave_table = _ames_dust_climatology(
            args.ames_dust_reference, grid
        )
        base = dataclasses.replace(
            base,
            dust_climatology_ls_deg=jnp.asarray(dust_ls),
            dust_visible_climatology=jnp.asarray(dust_visible_table),
            dust_longwave_climatology=jnp.asarray(dust_longwave_table),
        )
    steps_total = round(args.sols * base.rotation_period_s / args.dt)
    steps_chunk = max(1, round(args.chunk_sols * base.rotation_period_s / args.dt))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sols": args.sols, "dt_seconds": args.dt,
                "truncation": args.truncation, "layers": args.layers, "runs": {}}
    import src
    source_package=Path(src.__file__).resolve().parent
    source_files=sorted(source_package.rglob('*.py'))
    source_files+=sorted((source_package/'framework/physics').glob('*.npz'))
    source_files+=sorted((source_package/'framework/physics/licenses').glob('*.txt'))
    source_content={'package/src/'+str(path.relative_to(source_package)):path.read_bytes() for path in source_files}
    source_content['scripts/'+Path(__file__).name]=Path(__file__).read_bytes()
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in source_content.items()}
    runtime=dict(python=sys.version,jax=jax.__version__,numpy=np.__version__,
        devices=[dict(platform=device.platform,kind=device.device_kind,id=device.id) for device in jax.devices()],
        xla_flags=os.environ.get('XLA_FLAGS',''))
    runtime_path=args.output_dir/f'runtime-{time.time_ns()}.json'
    if Path('/opt/environment.txt').is_file():
        runtime['installed_packages']=Path('/opt/environment.txt').read_text()
    runtime_path.write_text(json.dumps(runtime,indent=2)+'\n')
    manifest['runtime']=str(runtime_path)
    for name in configs:
        forcing = base if reference else dataclasses.replace(base, **ABLATIONS[name])
        root = args.output_dir / name
        restart_path = root / "restart.npz"
        progress_path = root / "progress.json"
        diagnostics_path = root / "checkpoint_diagnostics.csv"
        config_path = root / "config.json"
        config = dict(truncation=args.truncation, layers=args.layers, dt=args.dt,
                      initial_ls=args.initial_ls, diurnal=forcing.diurnal,
                      dust_visible=args.dust_visible, dust_longwave=args.dust_longwave,
                      hyperdiffusion_tau_sols=args.hyperdiffusion_tau_sols,
                      solar_slant_path_enabled=True,
                      co2_lw_scale=args.co2_lw_scale,
                      dust_lw_scale=args.dust_lw_scale,
                      surface_exchange_multiplier=args.surface_exchange_multiplier,
                      physics=name, float64=True,
                      source_sha256=source_hashes,
                      co2_forcing=None if reference else dataclasses.asdict(co2_forcing(energy_limited=True)),
                      co2_frost_radiative_properties=forcing.co2_frost_radiative_properties,
                      surface_sha256=None if reference else hashlib.sha256(args.surface_properties.read_bytes()).hexdigest(),
                      ames_dust_reference=(str(args.ames_dust_reference) if args.ames_dust_reference else None),
                      ames_dust_sha256=(hashlib.sha256(args.ames_dust_reference.read_bytes()).hexdigest()
                                        if args.ames_dust_reference else None))
        if reference:
            config.update(reference_input_sha256=input_hashes,body=dataclasses.asdict(body),
                initial_dry_mass_parameter_pa=705.,initial_pressure_scheme='released-hydro-eq',reference_protocol='present-Mars-clouds-v3.2',
                radiation_mode='12band',radiation_update_seconds=8*args.dt,
                reference_release_commit='ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26',
                reference_build_blob='83b611e6527cf26bef3ec0064c05aebe0bb0d080',
                initial_air_temperature_k=args.initial_air_temperature,isothermal_atmosphere=args.isothermal_atmosphere,
                initial_surface_profile='released-equilibrated-latitude',
                sigma_boundaries=sigma.tolist())
            config.update(alternate_radiation_metadata)
            if args.scale_surface_ir_flux: config['reference_radiation_controls']=dict(scale_surface_flux=True)
            if solar_constant is not None:
                config.setdefault('reference_radiation_controls',{})['solar_constant_w_m2']=solar_constant
            if args.linear_k_interpolation:
                config.setdefault('reference_radiation_controls',{})['linear_k_interpolation']=True
            if args.shortwave_ramp_sols>0.:
                config.setdefault('reference_radiation_controls',{})['shortwave_ramp_sols']=args.shortwave_ramp_sols
            if args.fixed_radiation_date>=0:
                config.setdefault('reference_radiation_controls',{})['fixed_calendar_date']=args.fixed_radiation_date
            if soil_controls: config['reference_soil_controls']=soil_controls
            if surface_map_controls: config['reference_surface_map_controls']=surface_map_controls
            if initial_surface_controls: config['reference_initial_surface_controls']=initial_surface_controls
            if not args.equilibrated_surface:
                config.update(initial_surface_profile='uniform',initial_soil_temperature_k=args.soil_temperature)
            if not all(co2_controls.values()): config['co2_process_switches']=co2_controls
            if not all(surface_controls.values()): config['surface_process_switches']=surface_controls
            if process_controls: config['reference_process_controls']=process_controls
            if not args.do_mars_surface: config['reference_protocol']='released-no-surface-v3.2'
            elif not args.moment_water:
                config.update(reference_protocol='released-dust-without-water-v3.2',moment_water=False,moment_dust=True)
        if restart_path.exists():
            if not args.resume:
                raise ValueError(f"{restart_path} exists; use --resume or a new output directory")
            existing_config = (
                json.loads(config_path.read_text()) if config_path.exists() else {}
            )
            # Restarts produced before this explicit calibration control used
            # the mathematically identical multiplier of one.
            existing_config.setdefault("surface_exchange_multiplier", 1.0)
            if existing_config != config:
                raise ValueError("Restart configuration is missing or differs; use a new output directory")
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "executed_source.zip").exists():
            with zipfile.ZipFile(root / "executed_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for filename, content in source_content.items():
                    archive.writestr(filename, content)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        state = load_restart(restart_path) if args.resume and restart_path.exists() else None
        completed = (round(float(state.dynamics.sim_time) / float(
            specs.nondimensionalize(args.dt * scales.units.second)
        )) if state is not None else 0)
        invocation_start_step = completed
        invocation_start_time = time.perf_counter()
        if completed >= steps_total:
            manifest["runs"][name] = {
                "steps": completed, "restart": str(restart_path),
                "netcdf": str(root / "maps.nc"), "status": "already_complete",
                "checkpoint_diagnostics": str(diagnostics_path),
            }
            continue
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * scales.units.second))

        def diagnostic_record(checkpoint_state):
            record = _checkpoint_diagnostics(
                checkpoint_state, coords, specs, args.dt,body=body
            )
            if reference:
                from src.framework.physics.dinosaur_physics import water_inventory,exchange_energy
                from src.framework.physics.water_moments import LATENT_HEAT
                record['total_water_mass_kg']=float(water_inventory(checkpoint_state,coords,specs,body))
                record['exchange_energy_j']=float(exchange_energy(checkpoint_state,coords,specs,body,
                    soil_volumetric_heat_capacity=reference_options['soil_volumetric_heat_capacity'],
                    water_vapor_tracer='water_vapor' if reference_options.get('moment_water',False) else None,
                    water_latent_heat_j_kg=LATENT_HEAT))
                pressure=np.exp(np.asarray(grid.to_nodal(checkpoint_state.dynamics.log_surface_pressure)))[0]*float(
                    specs.dimensionalize(1.,scales.units.pascal).magnitude)
                dust=sum((np.asarray(grid.to_nodal(checkpoint_state.dynamics.tracers[key]))
                    for key in ('dust_mass','water_core_mass') if key in checkpoint_state.dynamics.tracers),
                    np.zeros((args.layers,*grid.nodal_shape)))
                dust=np.sum(dust*np.diff(coords.vertical.boundaries)[:,None,None]*pressure[None]/body.gravity_m_s2,axis=0)
                if checkpoint_state.auxiliary_fields is not None and 'dust_surface_kg_m2' in checkpoint_state.auxiliary_fields:
                    dust=dust+np.asarray(checkpoint_state.auxiliary_fields['dust_surface_kg_m2'])[0]
                weights=np.asarray(grid.quadrature_weights)
                record['total_dust_mass_kg']=float(np.sum(dust*weights)/np.sum(weights)*body.surface_area_m2)
            if args.ames_dust_reference is not None:
                dust_visible, dust_longwave = dust_optical_depths(
                    checkpoint_state.dynamics.sim_time * time_scale_s, forcing
                )
                dust_visible = np.asarray(dust_visible)
                dust_longwave = np.asarray(dust_longwave)
                normalized_weights = np.asarray(
                    grid.quadrature_weights, dtype=np.float64
                ).copy()
                normalized_weights /= normalized_weights.sum()
                record["mean_dust_visible_optical_depth"] = float(
                    np.sum(dust_visible * normalized_weights)
                )
                record["mean_dust_longwave_optical_depth"] = float(
                    np.sum(dust_longwave * normalized_weights)
                )
            return record

        if state is None and not diagnostics_path.exists():
            # Establish the exact denominator for all conservation-drift claims.
            # This mirrors run_maps' MOLA-aware resting-state initialization.
            elevation_nodal_m = regrid_to_nodal(coords,mola_path=args.mola_path)
            initial_dynamics = reference_rest_state(coords,specs,body,elevation_nodal_m,
                temperature_k=args.initial_air_temperature,isothermal=args.isothermal_atmosphere) if reference else initial_rest_state(
                coords,
                specs,
                body,
                elevation_nodal_m,
            )
            initial_dynamics = dataclasses.replace(initial_dynamics, sim_time=0.0)
            if reference:
                from src.framework.physics.dinosaur_physics import initial_state as reference_initial_state
                initial_options={key:value for key,value in reference_options.items() if key.startswith('initial_')}
                state=reference_initial_state(initial_dynamics,coords,specs,temperature_k=170.,
                    moment_water=reference_options.get('moment_water',False),
                    moment_dust=reference_options.get('moment_dust',False),
                    dust_background_options=reference_options.get('dust_background_options'),
                    radiation_update_seconds=reference_options.get('radiation_update_seconds'),**initial_options)
                surface=(reference_surface_temperature_k(jnp.asarray(grid.latitudes),
                    equilibrated=args.equilibrated_surface,soil_temperature_k=args.soil_temperature)[None,None,:]
                    if args.do_mars_surface else jnp.full((1,1,len(grid.latitudes)),170.))
                surface=surface*float(specs.nondimensionalize(1.*scales.units.kelvin))
                surface=jnp.broadcast_to(surface,state.surface_temperature.shape)
                state=state._replace(surface_temperature=surface,
                    ground_temperature=jnp.broadcast_to(surface,state.ground_temperature.shape))
                initial_state=state
            else:
                initial_state = initial_column_state(initial_dynamics,coords,body.reference_temperature_k,
                    specs,forcing=forcing)
            _append_checkpoint_diagnostics(
                diagnostics_path,
                diagnostic_record(initial_state),
            )
        elif state is not None:
            # A previous invocation may have saved its restart immediately
            # before diagnostic serialization was interrupted. Restore that
            # exact checkpoint row before advancing further.
            _append_checkpoint_diagnostics(
                diagnostics_path,
                diagnostic_record(state),
            )

        def checkpoint(done, _total, checkpoint_state, _coords, _specs):
            absolute_step = completed + done
            if not all(
                np.isfinite(np.asarray(leaf)).all()
                for leaf in jax.tree_util.tree_leaves(checkpoint_state)
            ):
                raise FloatingPointError(
                    f"{name} became non-finite before step {absolute_step}; "
                    "the last saved restart remains valid"
                )
            save_restart(checkpoint_state, restart_path)
            save_restart(
                checkpoint_state,
                checkpoint_dir / f"restart_{absolute_step:09d}.npz",
            )
            if reference:
                from src.celestials.planets.mars.maps import save_comparison_netcdf
                save_comparison_netcdf(checkpoint_state,coords,specs,body,forcing,
                    checkpoint_dir/f"state_{absolute_step:09d}.nc")
            progress_path.write_text(
                json.dumps({"completed_steps": absolute_step}) + "\n"
            )
            record = diagnostic_record(checkpoint_state)
            _append_checkpoint_diagnostics(diagnostics_path, record)
            elapsed_seconds = float(checkpoint_state.dynamics.sim_time) * time_scale_s
            ls_deg = float(np.degrees(
                float(_true_anomaly(elapsed_seconds, forcing)) + forcing.ls_perihelion_rad
            ) % 360.0)
            print(
                f"{name}: {absolute_step}/{steps_total} steps; Ls={ls_deg:.3f}",
                flush=True,
            )
            if absolute_step < steps_total and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)

        remaining = steps_total - completed
        fields, state = run_maps(
            body=body,mola_path=args.mola_path,p0_pa=705. if reference else None,
            sigma_boundaries=sigma,
            truncation=args.truncation, n_layers=args.layers,
            dt_seconds=args.dt, n_steps=remaining, forcing=forcing,
            co2_forcing=None if reference else co2_forcing(energy_limited=True),
            physics_backend='reference' if reference else 'standard',
            radiation_component=radiation,reference_options=reference_options,
            initial_state=state, return_final_state=True,
            diagnostic_callback=checkpoint,
            progress_chunk_steps=steps_chunk,
            hyperdiffusion_tau_seconds=(
                args.hyperdiffusion_tau_sols * body.solar_day_seconds
            ),
        )
        completed = steps_total
        save_netcdf(fields, root / f"sample_{completed:09d}.nc")
        nc = save_netcdf(fields, root / "maps.nc")
        pngs = plot_maps(fields, root, prefix=name)
        elapsed_seconds = time.perf_counter() - invocation_start_time
        invocation_steps = completed - invocation_start_step
        simulated_sols = invocation_steps * args.dt / base.rotation_period_s
        manifest["runs"][name] = {
            "steps": completed, "restart": str(restart_path), "netcdf": str(nc),
            "plots": [str(path) for path in pngs], "physics": fields.physics,
            "checkpoint_diagnostics": str(diagnostics_path),
            "invocation_start_step": invocation_start_step,
            "invocation_steps": invocation_steps,
            "invocation_elapsed_seconds": elapsed_seconds,
            "simulated_sols_per_wall_hour": (
                simulated_sols * 3600.0 / elapsed_seconds
                if elapsed_seconds > 0.0 else None
            ),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
