#!/usr/bin/env python3
"""Convert frozen native MACDA windows into coupled-experiment restarts/targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from neural_temp_data import read_run, sha256
from cache_neural_temp import initialize
from run_differentiable_experiments import d, p
from src.framework.gcm.restart import save_restart


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-split", type=int, default=4,
                    help="Predeclared deterministic pilot subset; not the full native campaign")
    args = ap.parse_args()
    if args.per_split < 1:
        ap.error("--per-split must be positive")
    contract, manifest = read_run(args.native_run)
    if sha256(contract["terrain_path"]) != contract["terrain_sha256"]:
        raise ValueError("Terrain checksum changed")
    args.output.mkdir(parents=True, exist_ok=False)
    windows = []
    for split in ("train", "validation", "test"):
        pool = sorted([r for r in manifest["starts"] if r["split"] == split], key=lambda r: (r["quadrant"], r["year"], r["id"]))
        if len(pool) < args.per_split:
            raise ValueError(f"Not enough {split} windows")
        # Evenly sample the ordered seasonal pool, without inspecting forecast errors.
        selected = [pool[i] for i in np.linspace(0, len(pool)-1, args.per_split, dtype=int)]
        for record in selected:
            path = args.native_run / "native" / f"{record['id']}.nc"
            if sha256(path) != record["native_sha256"]:
                raise ValueError("Native checksum mismatch")
            with xr.open_dataset(path, decode_times=False) as ds:
                ds = ds.load()
            _, _, forcing, state = initialize(ds, record, contract)
            stem = record["id"]
            restart = args.output / f"{stem}-restart.npz"
            target = args.output / f"{stem}-target.npz"
            force = args.output / f"{stem}-forcing.npz"
            save_restart(state, restart)
            fields = np.stack([np.asarray(ds[name].isel(time=13).transpose("lev", "lon", "lat"))
                               for name in ("temp", "uwind", "vwind")])[None]
            np.savez_compressed(target, seconds=[contract["lead_seconds"]], fields=fields)
            np.savez_compressed(force, init_orbital_angle_rad=forcing.init_orbital_angle_rad,
                                dust_visible=forcing.dust_visible_optical_depth,
                                dust_longwave=forcing.dust_longwave_optical_depth)
            start = record["start_sol"] * d.MARS_BODY_3D.rotation_period_s
            windows.append(dict(id=stem, split=split, block=record["block"],
                source=contract["source_url"] + "@" + contract["source_revision"],
                reference_kind="reanalysis", start_seconds=start,
                end_seconds=start+contract["lead_seconds"], restart=restart.name,
                target=target.name, forcing=force.name, restart_sha256=sha256(restart),
                target_sha256=sha256(target), forcing_sha256=sha256(force)))
    p.dump(args.output / "windows.json", dict(schema_version=1, windows=windows,
        dt_seconds=contract["dt_seconds"], native_contract=contract,
        limitation="Prespecified seasonal pilot, not full-coverage held-out climatology"))
    print(json.dumps({"windows": len(windows), "dt_seconds": contract["dt_seconds"],
                      "manifest": str(args.output / "windows.json")}, indent=2))


if __name__ == "__main__":
    main()
