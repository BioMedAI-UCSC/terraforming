#!/usr/bin/env python3
"""Archive historical saved results without converting or relabelling their physics."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Archive output must be outside the source run")
    output.mkdir(parents=True, exist_ok=True)
    target = output / (source.name + ".tar.gz")
    manifest_path = output / (source.name + "-manifest.json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Use a new output directory; existing archives are immutable")
    environment = json.loads((source / "environment.json").read_text())
    records, arrays = [], 0
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Historical bundle must not follow symbolic links")
    for path in files:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for name in data.files:
                    value = data[name]
                    if np.issubdtype(value.dtype, np.number):
                        if not np.isfinite(value).all():
                            raise ValueError(f"Nonfinite saved array: {path.name}/{name}")
                        arrays += 1
        records.append(dict(path=str(path.relative_to(source)), bytes=path.stat().st_size, sha256=digest(path)))
    temporary = target.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=source.name + "/" + str(path.relative_to(source)), recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        for row in records:
            with archive.extractfile(source.name + "/" + row["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"Archive verification failed: {row['path']}")
    temporary.rename(target)
    manifest = dict(status="historical_saved_results_not_current_physics", archive=target.name,
        archive_sha256=digest(target), bytes=target.stat().st_size,
        recorded_git_commit=environment["git_commit"], recorded_git_status=environment["git_status"],
        original_gpu_model=None, numeric_saved_arrays_checked=arrays, files=records,
        limitations=["Original GPU model was not recorded; cuda:0 is not a hardware identity.",
            "These are the available saved checkpoints and final dimensional maps, not every integration step.",
            "Nondimensional restarts require their original source/unit contract; do not load them as current-physics states.",
            "Finite saved arrays and constrained inventories do not prove equilibrium, raw budget closure, or observational accuracy.",
            "Original dirty patches and complete input hashes cannot be reconstructed from absent records."])
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = archive_run(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("archive", "archive_sha256", "bytes", "numeric_saved_arrays_checked")}))
