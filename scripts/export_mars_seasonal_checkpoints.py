#!/usr/bin/env python3
"""Export comparison NetCDFs from second-year restart checkpoints."""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np

from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import (
    forcing_with_surface_properties,
    run_maps,
    save_netcdf,
)
from src.framework.gcm._dinosaur import jax, jnp, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.restart import load_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import _true_anomaly, mean_anomaly_for_ls
from run_gcm3d_ablation import ABLATIONS, _ames_dust_climatology


def circular_distance(values, targets):
    return min(abs((values - target + 180.0) % 360.0 - 180.0) for target in targets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("ames_reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0,90,180,270")
    parser.add_argument("--half-width", type=float, default=6.0)
    parser.add_argument("--evaluation-start-sol", type=float, default=668.0)
    parser.add_argument("--dt", type=float, default=300.0)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    coords = coordinate_system("T21", 12)
    specs = physics_specs(MARS_BODY_3D)
    base = radiative_forcing(
        diurnal=False, co2_radiation_enabled=True,
        dust_visible_optical_depth=0.3, dust_longwave_optical_depth=0.1,
    )
    base = dataclasses.replace(
        base, init_orbital_angle_rad=mean_anomaly_for_ls(0.0, base)
    )
    base = forcing_with_surface_properties(
        base, coords.horizontal, args.surface_properties
    )
    dust_ls, visible, longwave = _ames_dust_climatology(
        args.ames_reference, coords.horizontal
    )
    forcing = dataclasses.replace(
        base, **ABLATIONS["convection"],
        dust_climatology_ls_deg=jnp.asarray(dust_ls),
        dust_visible_climatology=jnp.asarray(visible),
        dust_longwave_climatology=jnp.asarray(longwave),
    )
    targets = [float(value) % 360.0 for value in args.targets.split(",")]
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * scales.units.second))
    selected = []
    for path in sorted(args.checkpoints.glob("restart_*.npz")):
        state = load_restart(path)
        elapsed_seconds = float(state.dynamics.sim_time) * time_scale_s
        elapsed_sols = elapsed_seconds / MARS_BODY_3D.rotation_period_s
        ls_deg = float(np.degrees(
            float(_true_anomaly(elapsed_seconds, forcing)) + forcing.ls_perihelion_rad
        ) % 360.0)
        if elapsed_sols >= args.evaluation_start_sol and circular_distance(ls_deg, targets) <= args.half_width:
            selected.append((path, state, elapsed_sols, ls_deg))
    if not selected:
        parser.error("no evaluation-year checkpoints fall inside target windows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, state, elapsed_sols, ls_deg in selected:
        fields = run_maps(
            truncation="T21", n_layers=12, dt_seconds=args.dt, n_steps=1,
            forcing=forcing, co2_forcing=co2_forcing(energy_limited=True),
            initial_state=state,
        )
        output = args.output_dir / f"sample_{int(round(elapsed_sols * 1000)):09d}.nc"
        save_netcdf(fields, output)
        print(f"{path.name}: sol={elapsed_sols:.3f}, Ls={ls_deg:.3f} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
