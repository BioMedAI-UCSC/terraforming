#!/usr/bin/env python3
"""Run restartable Mars-year soil spin-up and incremental PBL ablations.

The default is intentionally a climate experiment (668 sols), not the short map
demo. Use ``--sols`` for smoke tests. Each chunk writes a versioned restart; a
stopped run resumes with ``--resume`` without reinitialising the soil column.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
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

import numpy as np
from src.framework.gcm._dinosaur import jax

from src.framework.gcm.coordinates import coordinate_system
from src.celestials.planets.mars.maps import forcing_with_surface_properties, plot_maps, run_maps, save_netcdf
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.framework.gcm.restart import load_restart, save_restart


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gcm3d_ablation"))
    parser.add_argument("--config", choices=["all", *ABLATIONS], default="all")
    parser.add_argument("--truncation", default="T42")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dt", type=float, default=450.0)
    parser.add_argument("--sols", type=float, default=668.0)
    parser.add_argument("--chunk-sols", type=float, default=10.0)
    parser.add_argument(
        "--cooldown-seconds", type=float, default=5.0,
        help="idle time after every checkpoint to limit sustained laptop load",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dust-visible", type=float, default=0.3)
    parser.add_argument("--dust-longwave", type=float, default=0.1)
    parser.add_argument(
        "--laptop", action="store_true",
        help="use T21/8 levels, 600 s steps, 5-sol chunks and 15 s cooldowns",
    )
    args = parser.parse_args()
    if args.laptop:
        args.truncation = "T21"
        args.layers = 8
        args.dt = 600.0
        args.chunk_sols = min(args.chunk_sols, 5.0)
        args.cooldown_seconds = max(args.cooldown_seconds, 15.0)
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")

    verified_dt = {"T42": 450.0, "T85": 300.0, "T106": 225.0, "T170": 150.0}
    if args.truncation in verified_dt and args.dt > verified_dt[args.truncation]:
        parser.error(
            f"--dt={args.dt:g} s exceeds the verified {args.truncation} limit "
            f"of {verified_dt[args.truncation]:g} s"
        )

    configs = list(ABLATIONS) if args.config == "all" else [args.config]
    grid = coordinate_system(args.truncation, args.layers).horizontal
    base = radiative_forcing(
        diurnal=False, co2_radiation_enabled=True,
        dust_visible_optical_depth=args.dust_visible,
        dust_longwave_optical_depth=args.dust_longwave,
    )
    base = forcing_with_surface_properties(base, grid, args.surface_properties)
    steps_total = round(args.sols * base.rotation_period_s / args.dt)
    steps_chunk = max(1, round(args.chunk_sols * base.rotation_period_s / args.dt))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sols": args.sols, "dt_seconds": args.dt,
                "truncation": args.truncation, "layers": args.layers, "runs": {}}
    for name in configs:
        forcing = dataclasses.replace(base, **ABLATIONS[name])
        root = args.output_dir / name
        restart_path = root / "restart.npz"
        progress_path = root / "progress.json"
        state = load_restart(restart_path) if args.resume and restart_path.exists() else None
        completed = (
            int(json.loads(progress_path.read_text())["completed_steps"])
            if state is not None and progress_path.exists() else 0
        )
        if completed >= steps_total:
            manifest["runs"][name] = {
                "steps": completed, "restart": str(restart_path),
                "netcdf": str(root / "maps.nc"), "status": "already_complete",
            }
            continue
        while completed < steps_total:
            count = min(steps_chunk, steps_total - completed)
            fields, state = run_maps(
                truncation=args.truncation, n_layers=args.layers,
                dt_seconds=args.dt, n_steps=count, forcing=forcing,
                co2_forcing=co2_forcing(energy_limited=True),
                initial_state=state, return_final_state=True,
            )
            finite = all(
                np.isfinite(np.asarray(leaf)).all()
                for leaf in jax.tree_util.tree_leaves(state)
            )
            if not finite:
                raise FloatingPointError(
                    f"{name} became non-finite before step {completed + count}; "
                    "the last saved restart remains valid"
                )
            completed += count
            save_restart(state, restart_path)
            progress_path.write_text(json.dumps({"completed_steps": completed}) + "\n")
            if completed < steps_total and args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
        nc = save_netcdf(fields, root / "maps.nc")
        pngs = plot_maps(fields, root, prefix=name)
        manifest["runs"][name] = {
            "steps": completed, "restart": str(restart_path), "netcdf": str(nc),
            "plots": [str(path) for path in pngs], "physics": fields.physics,
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
