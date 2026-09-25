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
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
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


def _environment() -> dict[str, object]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    packages = {}
    for name in ("jax", "jaxlib", "dinosaur", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "packages": packages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _checkpoint_diagnostics(state, coords, specs, dt_seconds: float) -> dict[str, float | int]:
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
        (coords.vertical.layers, 1, 1), MARS_BODY_3D.reference_temperature_k
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
        MARS_BODY_3D,
        np.full(coords.vertical.layers, MARS_BODY_3D.reference_temperature_k),
    )
    elapsed_seconds = float(state.dynamics.sim_time) / float(
        specs.nondimensionalize(1.0 * scales.units.second)
    )
    step = round(elapsed_seconds / dt_seconds)
    mean_pressure_pa = float(np.sum(surface_pressure_pa * normalized_weights))
    mean_ice_pa = float(np.sum(co2_ice_pa * normalized_weights))
    co2_mass_kg = (
        (mean_pressure_pa + mean_ice_pa)
        * 4.0 * np.pi * MARS_BODY_3D.radius_m**2
        / MARS_BODY_3D.gravity_m_s2
    )
    return {
        "step": step,
        "elapsed_sols": elapsed_seconds / MARS_BODY_3D.rotation_period_s,
        "mean_surface_pressure_pa": mean_pressure_pa,
        "mean_co2_ice_pa": mean_ice_pa,
        "atmosphere_plus_surface_reservoir_mass_kg": co2_mass_kg,
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


def _nonfinite_state_paths(state) -> list[str]:
    """Return readable pytree paths for leaves containing NaN or infinity."""
    failed = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if not np.isfinite(np.asarray(leaf)).all():
            failed.append(jax.tree_util.keystr(path) or "<root>")
    return failed


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gcm3d_ablation"))
    parser.add_argument("--config", choices=["all", *ABLATIONS], default="all")
    parser.add_argument("--truncation", default="T21")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dt", type=float, default=300.0)
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
    parser.add_argument(
        "--performance-mode", action="store_true",
        help=(
            "remove artificial cooldowns; checkpoint spacing remains controlled "
            "by --chunk-sols and timing includes restart/diagnostic writes"
        ),
    )
    parser.add_argument(
        "--precision", choices=("float64", "float32"), default="float64",
        help="JAX numerical precision for the experiment (default: float64)",
    )
    parser.add_argument(
        "--physics-evaluation", choices=("stage", "step"), default="stage",
        help=(
            "evaluate column physics at every IMEX stage (reference) or once "
            "per complete timestep and hold its tendency through the stages"
        ),
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
    if args.laptop:
        args.truncation = "T21"
        args.layers = 12
        args.dt = 300.0
        args.chunk_sols = min(args.chunk_sols, 5.0)
        args.cooldown_seconds = max(args.cooldown_seconds, 15.0)
    if args.performance_mode:
        args.cooldown_seconds = 0.0
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
    jax.config.update("jax_enable_x64", args.precision == "float64")

    verified_dt = {
        "T21": 675.0,
        "T42": 450.0,
        "T85": 300.0,
        "T106": 225.0,
        "T170": 150.0,
    }
    if args.truncation in verified_dt and args.dt > verified_dt[args.truncation]:
        parser.error(
            f"--dt={args.dt:g} s exceeds the verified {args.truncation} limit "
            f"of {verified_dt[args.truncation]:g} s"
        )

    configs = list(ABLATIONS) if args.config == "all" else [args.config]
    coords = coordinate_system(args.truncation, args.layers)
    grid = coords.horizontal
    specs = physics_specs(MARS_BODY_3D)
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
    base = dataclasses.replace(base, init_orbital_angle_rad=mean_anomaly_for_ls(
        np.deg2rad(args.initial_ls), base
    ))
    base = forcing_with_surface_properties(base, grid, args.surface_properties)
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
    (args.output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2) + "\n"
    )
    manifest = {"sols": args.sols, "dt_seconds": args.dt,
                "truncation": args.truncation, "layers": args.layers, "runs": {}}
    for name in configs:
        forcing = dataclasses.replace(base, **ABLATIONS[name])
        root = args.output_dir / name
        restart_path = root / "restart.npz"
        progress_path = root / "progress.json"
        diagnostics_path = root / "checkpoint_diagnostics.csv"
        config_path = root / "config.json"
        config = dict(truncation=args.truncation, layers=args.layers, dt=args.dt,
                      initial_ls=args.initial_ls, diurnal=args.diurnal,
                      dust_visible=args.dust_visible, dust_longwave=args.dust_longwave,
                      hyperdiffusion_tau_sols=args.hyperdiffusion_tau_sols,
                      solar_slant_path_enabled=True,
                      co2_lw_scale=args.co2_lw_scale,
                      dust_lw_scale=args.dust_lw_scale,
                      surface_exchange_multiplier=args.surface_exchange_multiplier,
                      physics=name, precision=args.precision,
                      physics_evaluation=args.physics_evaluation,
                      surface_sha256=hashlib.sha256(args.surface_properties.read_bytes()).hexdigest(),
                      ames_dust_reference=(str(args.ames_dust_reference) if args.ames_dust_reference else None),
                      ames_dust_sha256=(hashlib.sha256(args.ames_dust_reference.read_bytes()).hexdigest()
                                        if args.ames_dust_reference else None))
        if restart_path.exists():
            if not args.resume:
                raise ValueError(f"{restart_path} exists; use --resume or a new output directory")
            existing_config = (
                json.loads(config_path.read_text()) if config_path.exists() else {}
            )
            # Restarts produced before this explicit calibration control used
            # the mathematically identical multiplier of one.
            existing_config.setdefault("surface_exchange_multiplier", 1.0)
            if "float64" in existing_config and "precision" not in existing_config:
                existing_config["precision"] = (
                    "float64" if existing_config.pop("float64") else "float32"
                )
            existing_config.setdefault("physics_evaluation", "stage")
            if existing_config != config:
                raise ValueError("Restart configuration is missing or differs; use a new output directory")
        root.mkdir(parents=True, exist_ok=True)
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
                checkpoint_state, coords, specs, args.dt
            )
            elapsed_seconds = float(checkpoint_state.dynamics.sim_time) * time_scale_s
            record["solar_longitude_deg"] = float(np.degrees(
                float(_true_anomaly(elapsed_seconds, forcing))
                + forcing.ls_perihelion_rad
            ) % 360.0)
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
            elevation_nodal_m = regrid_to_nodal(coords)
            initial_dynamics = initial_rest_state(
                coords,
                specs,
                MARS_BODY_3D,
                elevation_nodal_m,
            )
            initial_dynamics = dataclasses.replace(initial_dynamics, sim_time=0.0)
            initial_state = initial_column_state(
                initial_dynamics,
                coords,
                MARS_BODY_3D.reference_temperature_k,
                specs,
                forcing=forcing,
            )
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
            nonfinite_paths = _nonfinite_state_paths(checkpoint_state)
            if nonfinite_paths:
                raise FloatingPointError(
                    f"{name} became non-finite before step {absolute_step}; "
                    f"affected state leaves: {', '.join(nonfinite_paths)}; "
                    "the last saved restart remains valid"
                )
            save_restart(checkpoint_state, restart_path)
            save_restart(
                checkpoint_state,
                checkpoint_dir / f"restart_{absolute_step:09d}.npz",
            )
            progress_path.write_text(
                json.dumps({"completed_steps": absolute_step}) + "\n"
            )
            record = diagnostic_record(checkpoint_state)
            _append_checkpoint_diagnostics(diagnostics_path, record)
            ls_deg = record["solar_longitude_deg"]
            print(
                f"{name}: {absolute_step}/{steps_total} steps; Ls={ls_deg:.3f}",
                flush=True,
            )
            if absolute_step < steps_total and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)

        remaining = steps_total - completed
        integration_started = time.perf_counter()
        fields, state = run_maps(
            truncation=args.truncation, n_layers=args.layers,
            dt_seconds=args.dt, n_steps=remaining, forcing=forcing,
            co2_forcing=co2_forcing(energy_limited=True),
            initial_state=state, return_final_state=True,
            diagnostic_callback=checkpoint,
            progress_chunk_steps=steps_chunk,
            hyperdiffusion_tau_seconds=(
                args.hyperdiffusion_tau_sols * MARS_BODY_3D.rotation_period_s
            ),
            physics_evaluation=args.physics_evaluation,
        )
        integration_seconds = time.perf_counter() - integration_started
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
            "integration_seconds": integration_seconds,
            "timed_region": (
                "run_maps including compilation, checkpoint synchronization, "
                "restart writes, diagnostics, and final field decoding; excludes "
                "final NetCDF and plots"
            ),
            "simulated_sols_per_wall_hour": (
                simulated_sols * 3600.0 / elapsed_seconds
                if elapsed_seconds > 0.0 else None
            ),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
