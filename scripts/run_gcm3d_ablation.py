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


def reference_configuration(grid,data_directory,dt_seconds):
    """Released present-Mars cloud switches from build_run/fms_mars_clouds_v3.2."""
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    from src.celestials.planets.mars.surface import reference_surface_fields_on_grid
    from src.celestials.planets.mars.boundary_inputs import polar_water_on_grid
    from src.celestials.planets.mars.dust_scenario import load_scenario_on_grid
    from src.framework.physics.soil_frost import subsurface_properties
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER
    data_directory=Path(data_directory)
    fields=reference_surface_fields_on_grid(grid,data_directory)
    diffusivity,capacity=subsurface_properties(jnp.asarray(grid.latitudes)[None,:],
        fields['thermal_inertia'],fields['grs_ice_latitude_deg'])
    scenario=load_scenario_on_grid(grid,data_directory/'DustScenario_MY30.nc')
    forcing=dataclasses.replace(reference_radiative_forcing(),albedo=jnp.asarray(fields['albedo']),
        emissivity=jnp.asarray(fields['emissivity']),co2_frost_radiative_properties=True,
        water_frost_radiative_properties=True,diurnal=True)
    radiation_interval=8*dt_seconds
    radiation=make_radiation_component(make_particle_provider(bands=12,
        dust_tracers=(('dust_mass','dust_number'),),
        water_moment_tracers=(WATER_CLOUD_TRACER,'water_core_mass','water_ice_number')),
        radiation_interval_seconds=radiation_interval,radiation_mode='12band')
    options=dict(soil_diffusivity=diffusivity,soil_volumetric_heat_capacity=capacity,
        radiation_update_seconds=radiation_interval,moment_water=True,
        initial_water_surface_kg_m2=polar_water_on_grid(grid,data_directory),
        initial_water_vapor_ratio=0.,initial_dust_mass_ratio=0.,initial_dust_number_per_kg=0.,
        rayleigh_options=dict(model_top=True),
        dust_background_options=dict(scenario=scenario,effective_radius_m=2e-6),
        moment_cloud_options=dict(microtimestep=30.,contact=.95,latent_heat=False))
    inputs=('mars_thermal_inertia8_2011.nc','mars_TES_albedo8_new.nc','mars_sfc_emissivity8.nc',
        'GRS_interp16.nc','npcflag8.nc','DustScenario_MY30.nc')
    hashes={name:hashlib.sha256((data_directory/name).read_bytes()).hexdigest() for name in inputs}
    return AMES_BODY_3D,forcing,radiation,options,hashes


def reference_rest_state(coords,specs,body,elevation_nodal_m):
    """Match the released cloud run's 705 Pa dry-mass target over terrain."""
    dynamics=initial_rest_state(coords,specs,body,elevation_nodal_m,t_ref_k=170.,p0_pa=705.)
    grid=coords.horizontal
    scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    pressure=jnp.exp(grid.to_nodal(dynamics.log_surface_pressure))*scale
    weights=jnp.asarray(grid.quadrature_weights)
    mean=jnp.sum(pressure*weights)/jnp.sum(weights)
    unit=jnp.zeros_like(dynamics.log_surface_pressure).at[...,0,0].set(1.)
    normalization=jnp.mean(grid.to_nodal(unit))
    corrected=dynamics.log_surface_pressure.at[...,0,0].add(jnp.log(705./mean)/normalization)
    return dataclasses.replace(dynamics,log_surface_pressure=corrected)


def reference_surface_temperature_k(latitude_rad):
    """Released mars_surface cold start with use_equilibrated_ts enabled."""
    latitude=jnp.asarray(latitude_rad)
    return jnp.where(latitude < -15.*jnp.pi/180.,
        165.+55.5*.5*(1.+jnp.cos((latitude+.261799)*180./75.)),
        167.+52.*jnp.cos((latitude+.261799)*90./105.))


def alternate_reference_radiation(coords,mode,profile=None,timescale=-4.):
    """Select a released alternate branch and return its restart identity."""
    if mode=='simple':
        if profile is not None: raise ValueError('simple radiation does not read an equilibrium profile')
        from src.framework.physics.simple_radiation import make_radiation_component
        return make_radiation_component(),dict(radiation_update_seconds=None),dict(
            radiation_mode='simple',radiation_update_seconds=None,simple_geometry_interval_seconds=1800.)
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
    parser.add_argument('--equilibrium-profile',type=Path,help='released teq/zzf NetCDF input for fixed or height relaxation')
    parser.add_argument('--thermal-timescale',type=float,default=-4.,help='negative values use 86400-second days; positive values use seconds; zero disables')
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
    if not reference and (args.reference_radiation!='spectral' or args.equilibrium_profile is not None or args.thermal_timescale!=-4.):
        parser.error('alternate reference radiation requires --config reference')
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
        body,base,radiation,reference_options,input_hashes=reference_configuration(grid,args.surface_properties,args.dt)
        input_hashes['mola']=hashlib.sha256(args.mola_path.read_bytes()).hexdigest()
        if args.reference_radiation!='spectral':
            radiation,overrides,alternate_radiation_metadata=alternate_reference_radiation(
                coords,args.reference_radiation,args.equilibrium_profile,args.thermal_timescale)
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
                initial_mean_surface_pressure_pa=705.,reference_protocol='present-Mars-clouds-v3.2',
                radiation_mode='12band',radiation_update_seconds=8*args.dt,
                reference_release_commit='ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26',
                reference_build_blob='83b611e6527cf26bef3ec0064c05aebe0bb0d080',
                initial_air_temperature_k=170.,initial_surface_profile='released-equilibrated-latitude',
                sigma_boundaries=sigma.tolist())
            config.update(alternate_radiation_metadata)
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
                record['total_water_mass_kg']=float(water_inventory(checkpoint_state,coords,specs,body))
                record['exchange_energy_j']=float(exchange_energy(checkpoint_state,coords,specs,body,
                    soil_volumetric_heat_capacity=reference_options['soil_volumetric_heat_capacity']))
                pressure=np.exp(np.asarray(grid.to_nodal(checkpoint_state.dynamics.log_surface_pressure)))[0]*float(
                    specs.dimensionalize(1.,scales.units.pascal).magnitude)
                dust=sum(np.asarray(grid.to_nodal(checkpoint_state.dynamics.tracers[key]))
                    for key in ('dust_mass','water_core_mass'))
                dust=np.sum(dust*np.diff(coords.vertical.boundaries)[:,None,None]*pressure[None]/body.gravity_m_s2,axis=0)
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
            initial_dynamics = reference_rest_state(coords,specs,body,elevation_nodal_m) if reference else initial_rest_state(
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
                    moment_water=True,radiation_update_seconds=8*args.dt,**initial_options)
                surface=reference_surface_temperature_k(jnp.asarray(grid.latitudes))[None,None,:]
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
