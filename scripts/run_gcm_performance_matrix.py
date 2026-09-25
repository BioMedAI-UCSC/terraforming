#!/usr/bin/env python3
"""Run and validate the T21 timestep/precision performance matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MARS_SOL_SECONDS = 88_775.244


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["configuration"]) for row in rows]
    positions = np.arange(len(rows))
    colors = ["#4c78a8" if row["status"] in ("reference", "pass") else "#e45756"
              for row in rows]
    numeric = lambda value: np.nan if value is None else float(value)
    integration = np.asarray([numeric(row["integration_seconds"]) for row in rows])
    end_to_end = np.asarray([numeric(row["end_to_end_seconds"]) for row in rows])
    throughput = np.asarray([numeric(row["integration_sols_per_wall_hour"]) for row in rows])
    speedup = np.asarray([numeric(row["integration_speedup_vs_reference"]) for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(max(11, 1.4 * len(rows)), 8.5))
    width = 0.38
    axes[0, 0].bar(positions - width / 2, integration, width, label="Integration", color=colors)
    axes[0, 0].bar(positions + width / 2, end_to_end, width, label="End to end",
                   color=colors, alpha=0.45)
    axes[0, 0].set_ylabel("Wall time (s)")
    axes[0, 0].set_title("Runtime for matched simulated duration")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(positions, throughput, color=colors)
    axes[0, 1].set_ylabel("Simulated sols per wall hour")
    axes[0, 1].set_title("Integration throughput (compile inclusive)")

    axes[1, 0].bar(positions, speedup, color=colors)
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("Speedup vs dt300 FP64 stage")
    axes[1, 0].set_title("Integration speedup")

    temperature = np.asarray([
        float(row["max_temperature_error_k"])
        if row["max_temperature_error_k"] is not None else 0.0 for row in rows
    ])
    pressure = np.asarray([
        100.0 * float(row["max_pressure_relative_error"])
        if row["max_pressure_relative_error"] is not None else 0.0 for row in rows
    ])
    axes[1, 1].bar(positions - width / 2, temperature, width,
                   label="Temperature error (K)", color="#f58518")
    axes[1, 1].bar(positions + width / 2, pressure, width,
                   label="Pressure error (%)", color="#54a24b")
    axes[1, 1].set_title("Maximum scalar diagnostic error")
    axes[1, 1].legend(frameon=False)

    for index, row in enumerate(rows):
        if row["status"] == "execution_failed":
            for axis in axes.flat:
                axis.text(index, 0.02, "FAILED", rotation=90, color="#e45756",
                          ha="center", va="bottom", transform=axis.get_xaxis_transform(),
                          fontsize=8, fontweight="bold")

    for axis in axes.flat:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mars GCM timestep and physics-evaluation performance", fontsize=14)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("--ames-dust-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/performance/timestep-matrix"))
    parser.add_argument("--sols", type=float, default=30.0)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[300.0, 450.0, 600.0, 675.0])
    parser.add_argument("--precisions", nargs="+", choices=("float64", "float32"), default=["float64"])
    parser.add_argument(
        "--physics-evaluations", nargs="+", choices=("stage", "step"),
        default=["stage", "step"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="retry configurations previously recorded in failure.json",
    )
    args = parser.parse_args()

    if args.sols <= 0 or args.layers <= 0:
        parser.error("sols and layers must be positive")
    if 300.0 not in args.timesteps or "float64" not in args.precisions:
        parser.error("the matrix must include the float64 300 s reference")
    for path in (args.surface_properties, args.ames_dust_reference):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep caches on the same writable filesystem as all other artifacts. This
    # also overrides macOS-specific /private/tmp paths inherited on Linux SSH hosts.
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / "matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / "xdg-cache"))
    runner = Path(__file__).with_name("run_gcm3d_ablation.py")
    comparator = Path(__file__).with_name("compare_gcm_performance_runs.py")
    records = []
    reference_dir = args.output_dir / "dt300-float64-stage"

    ordered = [(300.0, "float64", "stage")] + [
        (dt, precision, physics_evaluation)
        for physics_evaluation in args.physics_evaluations
        for precision in args.precisions
        for dt in args.timesteps
        if (dt, precision, physics_evaluation) != (300.0, "float64", "stage")
    ]
    for dt, precision, physics_evaluation in ordered:
        label = f"dt{dt:g}-{precision}-{physics_evaluation}"
        output = args.output_dir / label
        manifest = output / "manifest.json"
        failure = output / "failure.json"
        command = [
            sys.executable, str(runner), str(args.surface_properties),
            "--output-dir", str(output), "--config", "convection",
            "--truncation", "T21", "--layers", str(args.layers),
            "--dt", str(dt), "--initial-ls", "0", "--diurnal",
            "--sols", str(args.sols), "--performance-mode",
            "--precision", precision, "--physics-evaluation", physics_evaluation,
            "--ames-dust-reference",
            str(args.ames_dust_reference),
        ]
        if args.resume:
            command.append("--resume")
        execution_failure = None
        if not manifest.exists() and failure.exists() and not args.retry_failed:
            execution_failure = json.loads(failure.read_text())
        elif (
            not manifest.exists()
            and output.exists()
            and any(output.iterdir())
            and not args.resume
            and not args.retry_failed
        ):
            execution_failure = {
                "status": "execution_failed",
                "reason": "incomplete output from a previous invocation",
                "configuration": label,
                "command": command,
            }
            failure.write_text(json.dumps(execution_failure, indent=2) + "\n")
        elif not manifest.exists():
            if args.retry_failed and failure.exists():
                failure.unlink()
            started = time.perf_counter()
            returncode = run(command)
            if returncode:
                execution_failure = {
                    "status": "execution_failed",
                    "reason": f"runner exited with status {returncode}",
                    "returncode": returncode,
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    "configuration": label,
                    "command": command,
                }
                output.mkdir(parents=True, exist_ok=True)
                failure.write_text(json.dumps(execution_failure, indent=2) + "\n")

        if execution_failure is not None or not manifest.exists():
            records.append({
                "configuration": label,
                "dt_seconds": dt,
                "precision": precision,
                "physics_evaluation": physics_evaluation,
                "status": "execution_failed",
                "integration_seconds": None,
                "end_to_end_seconds": execution_failure.get("elapsed_wall_seconds"),
                "integration_sols_per_wall_hour": None,
                "end_to_end_sols_per_wall_hour": None,
                "integration_speedup_vs_reference": None,
                "max_temperature_error_k": None,
                "max_pressure_relative_error": None,
                "co2_drift_factor": None,
                "seasonal_peak_error_deg": None,
                "comparison": None,
            })
            continue

        comparison = None
        comparison_report = None
        status = "reference"
        if output != reference_dir:
            comparison = output / "comparison.json"
            result = subprocess.run([
                sys.executable, str(comparator),
                "--reference", str(reference_dir / "convection" / "checkpoint_diagnostics.csv"),
                "--candidate", str(output / "convection" / "checkpoint_diagnostics.csv"),
                "--output", str(comparison),
            ], check=False)
            status = "pass" if result.returncode == 0 else "fail"
            comparison_report = json.loads(comparison.read_text())
        timing = json.loads(manifest.read_text())["runs"]["convection"]
        integration_seconds = timing.get("integration_seconds")
        end_to_end_seconds = timing.get("invocation_elapsed_seconds")
        simulated_sols = timing["invocation_steps"] * dt / MARS_SOL_SECONDS
        checks = comparison_report["checks"] if comparison_report else {}
        records.append({
            "configuration": label,
            "dt_seconds": dt,
            "precision": precision,
            "physics_evaluation": physics_evaluation,
            "status": status,
            "integration_seconds": integration_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "integration_sols_per_wall_hour": (
                simulated_sols * 3600.0 / integration_seconds
            ),
            "end_to_end_sols_per_wall_hour": timing.get("simulated_sols_per_wall_hour"),
            "integration_speedup_vs_reference": None,
            "max_temperature_error_k": checks.get("mean_surface_temperature_k", {}).get("value"),
            "max_pressure_relative_error": checks.get("mean_surface_pressure_relative", {}).get("value"),
            "co2_drift_factor": checks.get("co2_mass_relative_drift_factor", {}).get("value"),
            "seasonal_peak_error_deg": checks.get("seasonal_peak_ls_deg", {}).get("value"),
            "comparison": str(comparison) if comparison else None,
        })

    if records[0]["status"] == "execution_failed":
        raise RuntimeError("the dt300 float64 stage reference failed; comparison is impossible")
    reference_seconds = float(records[0]["integration_seconds"])
    for record in records:
        if record["integration_seconds"] is not None:
            record["integration_speedup_vs_reference"] = (
                reference_seconds / float(record["integration_seconds"])
            )
    failed = sum(record["status"] == "execution_failed" for record in records)
    summary = {
        "status": "complete_with_failures" if failed else "complete",
        "reference": "dt300-float64-stage",
        "execution_failures": failed,
        "runs": records,
    }
    (args.output_dir / "matrix.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.output_dir / "performance.csv", records)
    plot_results(args.output_dir / "performance", records)
    print(json.dumps({
        "status": summary["status"],
        "execution_failures": failed,
        "output": str(args.output_dir / "matrix.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
