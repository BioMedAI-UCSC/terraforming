"""Validate a frozen input bundle, require CUDA, and preserve run provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

FILES = ("restart.npz", "target.nc", "surface.nc", "dust.nc", "mola.img")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(config, inputs):
    required = {"rollout_sols": 0.25, "expected_steps": 74,
                "dt_seconds": 300.0, "max_evaluations": 20}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Canonical experiment requires {key}={value}")
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name in FILES:
        if sha256(inputs / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment.json"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check input hashes/configuration without initializing JAX")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = validate(config, args.inputs)
    if args.validate_only:
        print(json.dumps({"status": "inputs_valid", "config": config, "manifest": manifest}))
        return 0
    # Never silently overwrite or mix traces from a previous attempt.
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_FLAGS", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    (args.output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "experiment.json").write_text(json.dumps(config, indent=2))
    if Path("/opt/environment.txt").exists():
        shutil.copyfile("/opt/environment.txt", args.output / "environment.txt")
    print(json.dumps({"phase": "initializing_gpu", "config": config}), flush=True)
    from . import driver
    steps = round(config["rollout_sols"] * driver.MARS_BODY_3D.rotation_period_s / config["dt_seconds"])
    if steps != config["expected_steps"]:
        raise ValueError(f"Expected 74 steps, got {steps}")
    sys.argv = ["mars-calibration", *[str(args.inputs / name) for name in FILES[:4]],
                "--output", str(args.output / "report.json"),
                "--target-index", str(config["target_index"]),
                "--rollout-sols", str(config["rollout_sols"]),
                "--dt", str(config["dt_seconds"]),
                "--max-evaluations", str(config["max_evaluations"]),
                "--initial", config["initial"], "--require-gpu"]
    try:
        return driver.main()
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"status": "error", "error": str(exc)}))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
