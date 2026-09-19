#!/usr/bin/env python3
"""Evaluate year-over-year convergence from Mars GCM checkpoint diagnostics."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from src.celestials.planets.mars.gcm import radiative_forcing
from src.framework.physics.gcm import _true_anomaly, mean_anomaly_for_ls


DEFAULT_FIELDS = {
    "mean_surface_pressure_pa": 5.0,
    "mean_co2_ice_pa": 5.0,
    "mean_surface_temperature_k": 1.0,
    "mean_deep_soil_temperature_k": 0.25,
}


def circular_distance(values, target):
    return np.abs((np.asarray(values) - target + 180.0) % 360.0 - 180.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sol", type=float, required=True)
    parser.add_argument("--targets", default="45,135,225,315")
    parser.add_argument("--half-width", type=float, default=5.0)
    args = parser.parse_args()

    with args.diagnostics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("diagnostic history is empty")
    numeric = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    if not all(np.isfinite(value).all() for value in numeric.values()):
        raise ValueError("diagnostic history contains non-finite values")

    forcing = radiative_forcing(diurnal=True, co2_radiation_enabled=True)
    forcing = dataclasses.replace(
        forcing, init_orbital_angle_rad=mean_anomaly_for_ls(0.0, forcing)
    )
    year_sols = forcing.orbital_period_s / forcing.rotation_period_s
    elapsed_s = numeric["elapsed_sols"] * forcing.rotation_period_s
    ls = np.mod(np.degrees(np.asarray(_true_anomaly(elapsed_s, forcing)))
                + np.degrees(forcing.ls_perihelion_rad), 360.0)
    first = (numeric["elapsed_sols"] >= args.start_sol) & (
        numeric["elapsed_sols"] <= args.start_sol + year_sols
    )
    second = (numeric["elapsed_sols"] >= args.start_sol + year_sols) & (
        numeric["elapsed_sols"] <= args.start_sol + 2.0 * year_sols
    )
    if not first.any() or not second.any():
        raise ValueError("diagnostics do not contain two complete evaluation years")

    report = {
        "status": "pass",
        "start_sol": args.start_sol,
        "orbital_year_sols": year_sols,
        "first_year_end_sol": args.start_sol + year_sols,
        "second_year_end_sol": args.start_sol + 2.0 * year_sols,
        "half_width_ls_deg": args.half_width,
        "thresholds": DEFAULT_FIELDS,
        "seasons": {},
        "limitations": [
            "Convergence uses global checkpoint diagnostics, not local weather fields.",
            "Five-degree windows reduce instantaneous weather and diurnal-phase noise.",
            "Passing demonstrates a repeatable seasonal cycle at this resolution; it does not establish observational accuracy.",
        ],
    }
    targets = [float(item) % 360.0 for item in args.targets.split(",")]
    for target in targets:
        masks = [
            first & (circular_distance(ls, target) <= args.half_width),
            second & (circular_distance(ls, target) <= args.half_width),
        ]
        if not all(mask.any() for mask in masks):
            raise ValueError(f"Ls={target:g}: incomplete seasonal window")
        season = {
            "first_sample_count": int(masks[0].sum()),
            "second_sample_count": int(masks[1].sum()),
            "first_mean_ls_deg": float(np.mean(ls[masks[0]])),
            "second_mean_ls_deg": float(np.mean(ls[masks[1]])),
            "fields": {},
        }
        for name, threshold in DEFAULT_FIELDS.items():
            means = [float(np.mean(numeric[name][mask])) for mask in masks]
            delta = means[1] - means[0]
            passed = abs(delta) <= threshold
            season["fields"][name] = {
                "first_year_mean": means[0],
                "second_year_mean": means[1],
                "delta": delta,
                "absolute_delta": abs(delta),
                "threshold": threshold,
                "pass": passed,
            }
            if not passed:
                report["status"] = "fail"
        report["seasons"][f"ls{int(round(target)):03d}"] = season

    initial_mass = numeric["atmosphere_plus_surface_reservoir_mass_kg"][0]
    report["co2_mass_conservation"] = {
        "initial_mass_kg": float(initial_mass),
        "final_mass_kg": float(numeric["atmosphere_plus_surface_reservoir_mass_kg"][-1]),
        "maximum_absolute_relative_drift": float(np.max(np.abs(
            numeric["atmosphere_plus_surface_reservoir_mass_kg"] / initial_mass - 1.0
        ))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
