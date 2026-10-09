#!/usr/bin/env python3
"""Native MACDA manifests and immutable, content-addressed experiment contracts."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

REVISION = "65a0bebd804b9c240752277e83f5737d58c6ee9c"
SOURCE = f"https://huggingface.co/datasets/ananyo01/ARCO-MACDA/resolve/{REVISION}/macda_combined.zarr"
SPLITS = {"train": [24, 25, 26, 27, 29, 30, 31], "validation": [32, 33], "test": [34, 35]}
VARIABLES = ["temp", "uwind", "vwind", "psurf", "tsurf", "coldust", "co2ice"]
UNITS = {"temp": {"K"}, "tsurf": {"K"}, "uwind": {"m s-1", "m/s"},
         "vwind": {"m s-1", "m/s"}, "psurf": {"Pa"},
         "coldust": {"1", "", "dimensionless"}, "co2ice": {"kg m-2", "kg/m2"}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def freeze(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"immutable contract mismatch: {path}")
    write_json(path, value)


def open_source():
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(SOURCE), consolidated=True, decode_times=False)


def validate_units(source):
    for name, allowed in UNITS.items():
        if name not in source or source[name].attrs.get("units") not in allowed:
            raise ValueError(f"missing variable or unsupported units: {name}")
    # ARCO encodes elapsed Martian sols with a misleading CF day epoch.
    units = source.time.attrs.get("units", "")
    if units != "sol" and not (units.startswith("days since") and source.time.attrs.get("calendar") == "none"):
        raise ValueError(f"unsupported native time units: {units}")
    if source.lev.attrs.get("units") not in ("1", "", "sigma_level"):
        raise ValueError("source lev must be dimensionless sigma")
    for name, units in (("Ls", {"degree", "degrees"}), ("MY_Ls", {"1"})):
        if name not in source or source[name].attrs.get("units") not in units:
            raise ValueError(f"unsupported metadata units: {name}")


def select_manifest(time, year, ls, *, seed=0, validator=None):
    """Scan actual coordinates, including unordered blocks; never bridge a gap.

    Four equally spaced seasonal strata per quadrant avoid taking the first four
    records. Seeded priorities inside each stratum are independent of errors.
    Temporal bootstrap blocks are Mars-year/season pairs fixed before fitting.
    """
    time, year, ls = [np.asarray(x, dtype=float) for x in (time, year, ls)]
    if not (time.ndim == 1 and time.shape == year.shape == ls.shape):
        raise ValueError("metadata coordinates must be aligned vectors")
    candidates, exclusions = {}, []
    for i in range(12, len(time) - 1):
        sl = slice(i - 12, i + 2)
        if not np.isfinite(np.concatenate([time[sl], year[sl], ls[sl]])).all():
            reason = "nonfinite_metadata"
        elif not np.all(year[sl] == year[i]):
            reason = "year_boundary"
        elif not np.allclose(np.diff(time[sl]), 1 / 12, atol=1e-8, rtol=0):
            reason = "native_cadence_gap_or_order"
        elif not (0 <= ls[i] < 360):
            reason = "invalid_ls"
        else:
            candidates.setdefault((int(year[i]), int(ls[i] // 22.5)), []).append(i)
            continue
        exclusions.append({"index": i, "reason": reason})
    rng, selected, deficits = np.random.default_rng(seed), [], []
    for split, years in SPLITS.items():
        for my in years:
            used = []
            for stratum in range(16):
                found = False
                for i in rng.permutation(candidates.get((my, stratum), [])):
                    i = int(i)
                    if any(abs(time[i] - t) < 2 - 1e-9 for t in used):
                        continue
                    if any(time[i-12] <= r["times"][-1] and time[i+1] >= r["times"][0] for r in selected):
                        exclusions.append({"index": i, "reason": "overlapping_history_target_interval"})
                        continue
                    record = {"id": f"my{my}_q{stratum // 4}_s{stratum % 4}",
                              "split": split, "year": my, "quadrant": stratum // 4,
                              "block": f"my{my}_q{stratum // 4}", "start_index": i,
                              "indices": list(range(i - 12, i + 2)),
                              "times": time[i - 12:i + 2].tolist(),
                              "start_sol": float(time[i]), "ls": float(ls[i]),
                              "provenance": "unknown: concatenated global attributes are not per-window evidence"}
                    try:
                        if validator:
                            validator(record)
                    except (ValueError, FloatingPointError) as exc:
                        exclusions.append({"index": i, "reason": str(exc)})
                        continue
                    selected.append(record)
                    used.append(time[i])
                    found = True
                    break
                if not found:
                    deficits.append({"split": split, "year": my, "season_stratum": stratum})
    return {"schema": 1, "seed": seed, "source_revision": REVISION,
            "splits": SPLITS, "excluded_years": [28], "starts": selected,
            "requested": {k: len(v) * 16 for k, v in SPLITS.items()},
            "achieved": {k: sum(r["split"] == k for r in selected) for k in SPLITS},
            "exclusions": exclusions, "deficits": deficits,
            "bootstrap_blocks": "Mars year × seasonal quadrant; resample whole blocks"}


def load_window(source, record):
    from src.framework.gcm._dinosaur import jax
    from stage_arco_macda import _to_dinosaur_grid
    jax.config.update("jax_enable_x64", True)
    validate_units(source)
    if len(record["indices"]) != 14 or not np.allclose(np.diff(record["times"]), 1/12, atol=1e-8, rtol=0):
        raise ValueError("window must contain 14 consecutive native samples")
    sample = source[VARIABLES + ["Ls", "MY_Ls"]].isel(time=record["indices"]).load()
    if not np.allclose(sample.time, record["times"], rtol=0, atol=1e-9):
        raise ValueError("source times changed")
    if not np.all(np.asarray(sample.MY_Ls) == record["year"]):
        raise ValueError("split/year mismatch")
    for name in VARIABLES:
        a = np.asarray(sample[name])
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite source: {name}")
        if name in ("psurf", "temp", "tsurf") and np.any(a <= 0):
            raise ValueError(f"nonpositive {name}")
        if name in ("coldust", "co2ice") and np.any(a < 0):
            raise ValueError(f"negative {name}")
    result = _to_dinosaur_grid(sample, "T21", 12).load()
    if any(not np.isfinite(result[v]).all() for v in VARIABLES):
        raise ValueError("nonfinite regridded window")
    result.time.attrs["long_name"] = "native Martian sols since MY24 start"
    return result


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    files = (sorted((root / "package/src").rglob("*.py"))
             + sorted((root / "scripts").glob("*neural_temp*.py"))
             + sorted((root / "package/src/framework/physics").glob("*.npz"))
             + [root / "scripts/stage_arco_macda.py"])
    return digest({str(p.relative_to(root)): sha256(p) for p in files})


def make_contract(manifest, terrain):
    from dataclasses import asdict
    from importlib.metadata import version
    from src.celestials.planets.mars.gcm import radiative_forcing, co2_forcing
    from neural_temp_model import SCHEMA
    from cache_neural_temp import physical_forcing
    base = radiative_forcing()
    lead = base.rotation_period_s / 12
    return {"schema": 1, "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "implementation_hash": implementation_hash(), "source_revision": REVISION,
            "source_url": SOURCE, "manifest_hash": digest(manifest),
            "citations": ["10.57967/hf/8771", "10.5285/cd037a9ea387438fabf4d674dbe53088"],
            "grid": "T21/L12", "sigma_centers": ((np.arange(12) + .5) / 12).tolist(),
            "precision": "float64", "lead_sols": 1 / 12, "lead_seconds": lead,
            "software": {name: version(name) for name in ("numpy", "torch", "jax", "jaxlib", "dinosaur", "xarray")},
            "steps": int(np.ceil(lead / 300)), "dt_seconds": lead / np.ceil(lead / 300),
            "terrain_path": str(Path(terrain).resolve()), "terrain_sha256": sha256(terrain),
            "forcing": asdict(physical_forcing()), "co2": asdict(co2_forcing()),
            "dust": "start coldust held fixed; visible=coldust, longwave=coldust/3",
            "clock": "MACDA epoch midnight; solver clock=(source_sol+0.5)*rotation_period; orbit anchored to start Ls",
            "initialization": "observed spectral T/u/v/log(ps); start-only surface/frost; soil initially uniform oldest history T, insulated bottom, one sol prescribed past surface conduction",
            "source_variables": VARIABLES, "feature_schema": SCHEMA,
            "fit": {"seed": 0, "epochs": 100, "patience": 10, "lr": .001, "weight_decay": .0001,
                    "clip": 1., "batch_starts": 8, "columns": 128, "ridge": .001},
            "criteria": {"relative_rmse_reduction": .10, "neural_beats_linear": True,
                         "paired_block_mse_interval_excludes_zero": True, "beats_no_gcm": True}}


def read_run(root):
    from importlib.metadata import version
    root = Path(root)
    c, m = [json.loads((root / f"{n}.json").read_text()) for n in ("contract", "manifest")]
    if c["manifest_hash"] != digest(m) or m["deficits"]:
        raise ValueError("manifest mismatch or incomplete coverage")
    if c["implementation_hash"] != implementation_hash():
        raise ValueError("scientific implementation changed; create a new run")
    if any(version(name) != pinned for name, pinned in c["software"].items()):
        raise ValueError("scientific software versions changed; create a new run")
    return c, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mola", type=Path, required=True)
    args = p.parse_args()
    if (args.run_dir / "contract.json").exists():
        c, _ = read_run(args.run_dir)
        if c["terrain_sha256"] != sha256(args.mola):
            raise ValueError("terrain mismatch")
        print("Frozen complete manifest already exists")
        return
    source = open_source()
    validate_units(source)
    def validate(record):
        window = load_window(source, record)
        path = args.run_dir / "native" / f"{record['id']}.nc"
        # Local staging is CPU-only; no GPUs are reserved during metadata I/O.
        atomic_bytes(path, window.to_netcdf())
        record["native_sha256"] = sha256(path)
    manifest = select_manifest(source.time.values, source.MY_Ls.values, source.Ls.values, validator=validate)
    write_json(args.run_dir / "coverage.json", manifest)
    if manifest["deficits"]:
        raise ValueError(f"coverage deficit: {manifest['deficits']}; see coverage.json")
    freeze(args.run_dir / "manifest.json", manifest)
    freeze(args.run_dir / "contract.json", make_contract(manifest, args.mola))
    print(f"Frozen native coverage: {manifest['achieved']}")


if __name__ == "__main__":
    main()
