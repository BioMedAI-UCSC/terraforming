#!/usr/bin/env python3
"""Compare second-year Dinosaur seasonal windows with staged Ames MGCM output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

from benchmark_mcd import interpolate_periodic, weighted_metrics

FIELDS = ("surface_pressure", "temperature", "wind_speed", "co2_ice")


def circular_distance(values, target):
    return np.abs((np.asarray(values) - target + 180.0) % 360.0 - 180.0)


def weighted_time_mean(dataset, weights):
    weights = xr.DataArray(np.asarray(weights), dims="sample")
    return dataset.weighted(weights).mean("sample", keep_attrs=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_samples", type=Path)
    parser.add_argument("ames_reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="45,135,225,315")
    parser.add_argument("--half-width", type=float, default=5.0)
    parser.add_argument("--evaluation-start-sol", type=float, default=668.0)
    args = parser.parse_args()

    paths = sorted(args.model_samples.glob("sample_*.nc"))
    samples = []
    for path in paths:
        with xr.open_dataset(path) as opened:
            if "surface_temperature" in opened:
                sols = float(opened.time.isel(time=0))
                ls = float(opened.solar_longitude.isel(time=0))
                sample = xr.Dataset(
                    {
                        "surface_pressure": opened.surface_pressure.isel(time=0),
                        "temperature": opened.surface_temperature.isel(time=0),
                        "u": opened.eastward_wind.isel(time=0, sigma=-1),
                        "v": opened.northward_wind.isel(time=0, sigma=-1),
                        "wind_speed": opened.wind_speed.isel(time=0, sigma=-1),
                        "co2_ice": opened.co2_frost.isel(time=0),
                    }
                ).load()
                insolation = "diurnal"
            else:
                sols = (
                    float(opened.attrs["n_steps"])
                    * float(opened.attrs["dt_seconds"]) / 88775.244
                )
                ls = float(opened.attrs["solar_longitude_deg"])
                sample = opened[list(FIELDS) + ["u", "v"]].load()
                insolation = str(opened.attrs.get("insolation_sampling", "unknown"))
            if sols >= args.evaluation_start_sol:
                sample = sample.expand_dims(sample=[len(samples)]).assign_coords(
                    ls=("sample", [ls]),
                    elapsed_sols=("sample", [sols]),
                )
                sample.attrs["insolation_sampling"] = insolation
                samples.append(sample)
    if not samples:
        parser.error("no model checkpoints exist in the evaluation year")
    model = xr.concat(samples, dim="sample")
    model_insolation = str(model.attrs.get("insolation_sampling", "unknown"))
    with xr.open_dataset(args.ames_reference) as opened:
        ames = opened.load()
    targets = [float(item) % 360.0 for item in args.targets.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "model_reference_comparison_not_observational_validation",
        "model": "Dinosaur/JAX T21L12, second simulated Mars year",
        "reference": ames.attrs.get("title", "NASA Ames MGCM"),
        "half_width_ls_deg": args.half_width,
        "seasons": {},
        "limitations": [
            (
                "Model seasonal means average instantaneous checkpoints under "
                f"{model_insolation} insolation; Ames fields are five-sol averages."
            ),
            "Wind compares each model's lowest layer; their effective heights are not identical.",
            "Ames dust opacity is prescribed to Dinosaur piecewise at checkpoint cadence, so dust is a matched boundary condition rather than an independent target.",
            "Ames MGCM and MCD are model references, not observational truth.",
        ],
    }
    for target in targets:
        model_mask = circular_distance(model.ls.values, target) <= args.half_width
        ames_mask = circular_distance(ames.ls.values, target) <= args.half_width
        if not model_mask.any() or not ames_mask.any():
            raise ValueError(f"Ls={target:g}: empty model or Ames seasonal window")
        model_window = model.isel(sample=np.flatnonzero(model_mask))
        ames_window = ames.isel(ls=np.flatnonzero(ames_mask)).rename(ls="sample")
        model_mean = model_window.mean("sample", keep_attrs=True)
        ames_mean = weighted_time_mean(
            ames_window, ames_window.average_duration_days.values
        )
        model_mean.attrs.update(
            solar_longitude_deg=target,
            temporal_sampling=f"mean of {int(model_mask.sum())} checkpoints within +/-{args.half_width:g} deg Ls",
            climate_status="second-year seasonal evaluation window",
        )
        regridded = interpolate_periodic(ames_mean[list(FIELDS)], model_mean)
        season_metrics = {}
        for name in FIELDS:
            season_metrics[name] = weighted_metrics(
                model_mean[name].values, regridded[name].values, model_mean.lat.values
            )
            season_metrics[name]["ames_area_mean"] = season_metrics[name].pop("mcd_area_mean")
        tag = f"ls{int(target):03d}"
        model_path = args.output_dir / f"model_{tag}.nc"
        ames_path = args.output_dir / f"ames_{tag}_on_model_grid.nc"
        model_mean.to_netcdf(model_path)
        regridded.to_netcdf(ames_path)
        report["seasons"][tag] = {
            "model_sample_count": int(model_mask.sum()),
            "ames_sample_count": int(ames_mask.sum()),
            "model_ls_deg": [float(x) for x in model_window.ls.values],
            "ames_ls_deg": [float(x) for x in ames_window.sample.values],
            "metrics": season_metrics,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "ames_regridded_sha256": hashlib.sha256(ames_path.read_bytes()).hexdigest(),
        }
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
