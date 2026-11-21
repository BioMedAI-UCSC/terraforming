#!/usr/bin/env python3
"""Compare global diagnostics at exactly shared physical checkpoint times."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

FIELDS = ("mean_surface_temperature_k", "mean_surface_pressure_pa", "mean_co2_ice_pa",
          "mean_co2_cloud_ice_pa", "mean_deep_soil_temperature_k", "max_wind_speed_ms")


def paired_diagnostics(first, first_dt, second, second_dt):
    def indexed(rows, dt):
        times = np.array([row["step"] * dt for row in rows])
        if not np.isfinite(times).all() or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("At least two increasing finite physical times required")
        return dict(zip(times, rows, strict=True))
    a, b = indexed(first, first_dt), indexed(second, second_dt)
    times = np.array(sorted(a.keys() & b.keys()))
    if len(times) < 2 or times[0] != 0 or times[-1] != max(a) or times[-1] != max(b):
        raise ValueError("Shared samples must span the full common trajectory")
    result = {}
    for field in FIELDS:
        difference = np.array([a[t][field] - b[t][field] for t in times])
        if not np.isfinite(difference).all():
            raise ValueError(f"Nonfinite {field}")
        result[field] = dict(time_weighted_rms_difference=float(np.sqrt(
            np.trapezoid(difference**2, times) / (times[-1] - times[0]))),
            maximum_absolute_difference=float(np.max(np.abs(difference))),
            final_difference=float(difference[-1]))
    return dict(first_dt_seconds=first_dt, second_dt_seconds=second_dt,
                common_samples=len(times), seconds=float(times[-1] - times[0]), fields=result)


def analyse(studies):
    runs = []
    for dt in (450, 300, 150):
        folder = studies / f"timestep-{dt}"
        config = json.loads((folder / "configuration.json").read_text())
        report = json.loads((folder / "report.json").read_text())
        if report["status"] != "finite_complete" or config["grid"]["dt_seconds"] != dt:
            raise ValueError(f"Incomplete or mismatched {folder}")
        rows = json.loads((folder / "diagnostics.json").read_text())
        if rows[-1]["step"] != config["grid"]["n_steps"]:
            raise ValueError(f"Incomplete diagnostic history: {folder}")
        with np.load(folder / "initial.npz", allow_pickle=False) as data:
            initial = {k: data[k].copy() for k in data.files}
        runs.append((dt, config, rows, initial))
    reference = runs[0]
    for dt, config, rows, initial in runs[1:]:
        grid = {k: v for k, v in config["grid"].items() if k not in ("dt_seconds", "n_steps")}
        reference_grid = {k: v for k, v in reference[1]["grid"].items() if k not in ("dt_seconds", "n_steps")}
        if grid != reference_grid:
            raise ValueError("Paired numerical controls must agree except timestep and step count")
        for key in ("forcing", "co2_forcing", "input_sha256"):
            if config[key] != reference[1][key]:
                raise ValueError(f"Changed {key} in {dt}-second case")
        if config["identity"]["source_sha256"] != reference[1]["identity"]["source_sha256"]:
            raise ValueError("Paired runs must use identical source bytes")
        if initial.keys() != reference[3].keys() or any(
                not np.array_equal(initial[k], reference[3][k]) for k in initial):
            raise ValueError("Paired runs must use identical initial states")
    return dict(status="comparison_complete", pairs=[paired_diagnostics(a[2], a[0], b[2], b[0])
                for a, b in itertools.combinations(runs, 2)],
                limitations="Global diagnostics at shared times, without interpolation. This does not measure local field error, equilibrium, or observational accuracy; no acceptance threshold is inferred from the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyse(args.studies), indent=2, allow_nan=False) + "\n")
