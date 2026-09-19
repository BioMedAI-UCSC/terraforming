#!/usr/bin/env python3
"""Compare gridded seasonal means from two consecutive simulated Mars years."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from benchmark_mcd import weighted_metrics


FIELDS = (
    "surface_pressure",
    "temperature",
    "u",
    "v",
    "wind_speed",
    "co2_ice",
)


def circular_distance(values, target):
    return np.abs((np.asarray(values) - target + 180.0) % 360.0 - 180.0)


def load_samples(directory: Path) -> xr.Dataset:
    samples = []
    for path in sorted(directory.glob("sample_*.nc")):
        with xr.open_dataset(path) as opened:
            sample = opened[list(FIELDS) + ["elevation"]].load()
            sample = sample.expand_dims(sample=[len(samples)]).assign_coords(
                ls=("sample", [float(opened.attrs["solar_longitude_deg"])]),
                elapsed_sols=(
                    "sample",
                    [
                        float(opened.attrs["n_steps"])
                        * float(opened.attrs["dt_seconds"])
                        / 88775.244
                    ],
                ),
            )
            samples.append(sample)
    if not samples:
        raise ValueError(f"no sample_*.nc files found in {directory}")
    return xr.concat(samples, dim="sample")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_year", type=Path)
    parser.add_argument("second_year", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="45,135,225,315")
    parser.add_argument("--half-width", type=float, default=5.0)
    args = parser.parse_args()

    first = load_samples(args.first_year)
    second = load_samples(args.second_year)
    if not np.array_equal(first.lat.values, second.lat.values) or not np.array_equal(
        first.lon.values, second.lon.values
    ):
        raise ValueError("first- and second-year samples use different grids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "diagnostic_year_over_year_field_comparison",
        "half_width_ls_deg": args.half_width,
        "seasons": {},
        "limitations": [
            "Metrics compare seasonal-window means of instantaneous five-sol checkpoints.",
            "Surface temperature and winds retain sampling sensitivity to local time and weather.",
            "Small year-over-year differences support repeatability; they do not establish observational accuracy.",
        ],
    }
    targets = [float(item) % 360.0 for item in args.targets.split(",")]
    for target in targets:
        masks = [
            circular_distance(first.ls.values, target) <= args.half_width,
            circular_distance(second.ls.values, target) <= args.half_width,
        ]
        if not all(mask.any() for mask in masks):
            raise ValueError(f"Ls={target:g}: empty first- or second-year window")
        windows = [
            dataset.isel(sample=np.flatnonzero(mask))
            for dataset, mask in zip((first, second), masks, strict=True)
        ]
        means = [window.mean("sample", keep_attrs=True) for window in windows]
        metrics = {}
        for name in FIELDS:
            field_metrics = weighted_metrics(
                means[1][name].values,
                means[0][name].values,
                means[0].lat.values,
            )
            field_metrics["second_year_area_mean"] = field_metrics.pop(
                "model_area_mean"
            )
            field_metrics["first_year_area_mean"] = field_metrics.pop(
                "mcd_area_mean"
            )
            reference_mean_absolute = weighted_metrics(
                np.abs(means[0][name].values),
                np.zeros_like(means[0][name].values),
                means[0].lat.values,
            )["model_area_mean"]
            field_metrics["first_year_area_mean_absolute"] = (
                reference_mean_absolute
            )
            field_metrics["relative_mae_percent_of_first_year_mean_absolute"] = (
                100.0 * field_metrics["mae"] / reference_mean_absolute
                if reference_mean_absolute > 0.0
                else None
            )
            metrics[name] = field_metrics

        tag = f"ls{int(round(target)):03d}"
        paths = [
            args.output_dir / f"first_year_{tag}.nc",
            args.output_dir / f"second_year_{tag}.nc",
        ]
        for mean, path in zip(means, paths, strict=True):
            mean.attrs.update(
                solar_longitude_window_center_deg=target,
                solar_longitude_half_width_deg=args.half_width,
            )
            mean.to_netcdf(path)
        report["seasons"][tag] = {
            "first_sample_count": int(masks[0].sum()),
            "second_sample_count": int(masks[1].sum()),
            "first_ls_deg": [float(value) for value in windows[0].ls.values],
            "second_ls_deg": [float(value) for value in windows[1].ls.values],
            "metrics_second_minus_first": metrics,
            "first_year_sha256": sha256(paths[0]),
            "second_year_sha256": sha256(paths[1]),
        }

    output = args.output_dir / "field_convergence.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
