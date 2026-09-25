#!/usr/bin/env python3
"""Compare a candidate GCM run with a reference and enforce numerical gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_GATES = {
    "mean_surface_temperature_k": 1.0,
    "mean_surface_pressure_relative": 0.01,
    "co2_mass_relative_drift": 1.0e-5,
    "seasonal_peak_ls_deg": 5.0,
}


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if len(rows) < 2:
        raise ValueError(f"{path} must contain initialization and later diagnostics")
    elapsed = np.asarray([row["elapsed_sols"] for row in rows])
    if not np.all(np.isfinite(elapsed)) or np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"{path} elapsed_sols must be finite and strictly increasing")
    for index, row in enumerate(rows):
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"{path} row {index + 2} contains nonfinite values")
    return rows


def interpolate(rows: list[dict[str, float]], times: np.ndarray, field: str) -> np.ndarray:
    source_t = np.asarray([row["elapsed_sols"] for row in rows])
    source_y = np.asarray([row[field] for row in rows])
    return np.interp(times, source_t, source_y)


def circular_distance_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def compare(reference, candidate, gates):
    required = {
        "elapsed_sols", "mean_surface_pressure_pa", "mean_surface_temperature_k",
        "atmosphere_plus_surface_reservoir_mass_kg",
    }
    missing = required - reference[0].keys() | required - candidate[0].keys()
    if missing:
        raise ValueError(f"diagnostic CSVs lack required columns: {sorted(missing)}")

    start = max(reference[0]["elapsed_sols"], candidate[0]["elapsed_sols"])
    stop = min(reference[-1]["elapsed_sols"], candidate[-1]["elapsed_sols"])
    if stop <= start:
        raise ValueError("reference and candidate have no overlapping coverage")
    # Use a shared interpolation grid rather than selecting exact checkpoint
    # timestamps. Float32 sim_time can differ from float64 by a few ulps at the
    # nominally identical final checkpoint, leaving only the initial sample.
    sample_count = max(2, min(len(reference), len(candidate)))
    candidate_times = np.linspace(start, stop, sample_count)

    ref_temp = interpolate(reference, candidate_times, "mean_surface_temperature_k")
    cand_temp = interpolate(candidate, candidate_times, "mean_surface_temperature_k")
    ref_pressure = interpolate(reference, candidate_times, "mean_surface_pressure_pa")
    cand_pressure = interpolate(candidate, candidate_times, "mean_surface_pressure_pa")

    temperature_error = float(np.max(np.abs(cand_temp - ref_temp)))
    pressure_error = float(np.max(
        np.abs(cand_pressure - ref_pressure) / np.maximum(np.abs(ref_pressure), 1e-12)
    ))

    mass_field = "atmosphere_plus_surface_reservoir_mass_kg"
    ref_mass = np.asarray([row[mass_field] for row in reference])
    cand_mass = np.asarray([row[mass_field] for row in candidate])
    ref_drift = float(np.max(np.abs(ref_mass / ref_mass[0] - 1.0)))
    cand_drift = float(np.max(np.abs(cand_mass / cand_mass[0] - 1.0)))
    drift_factor = cand_drift / max(ref_drift, np.finfo(float).eps)

    peak_phase_error = None
    if (
        stop - start >= 600.0
        and "solar_longitude_deg" in reference[0]
        and "solar_longitude_deg" in candidate[0]
    ):
        ref_ls_series = interpolate(reference, candidate_times, "solar_longitude_deg")
        cand_ls_series = interpolate(candidate, candidate_times, "solar_longitude_deg")
        ref_ls = ref_ls_series[int(np.argmax(ref_pressure))]
        cand_ls = cand_ls_series[int(np.argmax(cand_pressure))]
        peak_phase_error = circular_distance_degrees(float(ref_ls), float(cand_ls))

    checks = {
        "mean_surface_temperature_k": {
            "value": temperature_error,
            "limit": gates["mean_surface_temperature_k"],
            "pass": temperature_error <= gates["mean_surface_temperature_k"],
        },
        "mean_surface_pressure_relative": {
            "value": pressure_error,
            "limit": gates["mean_surface_pressure_relative"],
            "pass": pressure_error <= gates["mean_surface_pressure_relative"],
        },
        "co2_mass_relative_drift": {
            "value": cand_drift,
            "candidate_relative_drift": cand_drift,
            "reference_relative_drift": ref_drift,
            "candidate_over_reference": drift_factor,
            "limit": gates["co2_mass_relative_drift"],
            "pass": cand_drift <= gates["co2_mass_relative_drift"],
        },
    }
    if peak_phase_error is not None:
        checks["seasonal_peak_ls_deg"] = {
            "value": peak_phase_error,
            "limit": gates["seasonal_peak_ls_deg"],
            "pass": peak_phase_error <= gates["seasonal_peak_ls_deg"],
        }
    return {
        "status": "pass" if all(check["pass"] for check in checks.values()) else "fail",
        "overlap_sols": [float(start), float(stop)],
        "sample_count": int(candidate_times.size),
        "checks": checks,
        "limitations": [
            "These are numerical agreement gates, not observational error bars.",
            "Seasonal peak phase is evaluated only when at least 600 sols overlap.",
            "Scalar checkpoint diagnostics do not replace three-dimensional field comparison.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", type=Path, help="JSON overrides for numerical gates")
    args = parser.parse_args()

    gates = dict(DEFAULT_GATES)
    if args.gates:
        overrides = json.loads(args.gates.read_text())
        unknown = overrides.keys() - gates.keys()
        if unknown:
            parser.error(f"unknown gates: {sorted(unknown)}")
        gates.update(overrides)
    if any(not isinstance(value, (int, float)) or value <= 0 for value in gates.values()):
        parser.error("all gates must be positive numbers")

    report = compare(read_rows(args.reference), read_rows(args.candidate), gates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
