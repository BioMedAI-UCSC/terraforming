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
from pathlib import Path

from src.gcm3d.coordinates import coordinate_system
from src.gcm3d.maps import forcing_with_surface_properties, plot_maps, run_maps, save_netcdf
from src.gcm3d.physics import mars_co2_forcing, mars_radiative_forcing
from src.gcm3d.restart import load_restart, save_restart


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
    parser.add_argument("--dt", type=float, default=1800.0)
    parser.add_argument("--sols", type=float, default=668.0)
    parser.add_argument("--chunk-sols", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dust-visible", type=float, default=0.3)
    parser.add_argument("--dust-longwave", type=float, default=0.1)
    args = parser.parse_args()

    configs = list(ABLATIONS) if args.config == "all" else [args.config]
    grid = coordinate_system(args.truncation, args.layers).horizontal
    base = mars_radiative_forcing(
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
        while completed < steps_total:
            count = min(steps_chunk, steps_total - completed)
            fields, state = run_maps(
                truncation=args.truncation, n_layers=args.layers,
                dt_seconds=args.dt, n_steps=count, forcing=forcing,
                co2_forcing=mars_co2_forcing(energy_limited=True),
                initial_state=state, return_final_state=True,
            )
            completed += count
            save_restart(state, restart_path)
            progress_path.write_text(json.dumps({"completed_steps": completed}) + "\n")
        fields = dataclasses.replace(fields, n_steps=completed)
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
