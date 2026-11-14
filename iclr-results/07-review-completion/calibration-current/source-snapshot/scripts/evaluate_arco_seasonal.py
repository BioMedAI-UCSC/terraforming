#!/usr/bin/env python3
"""Evaluate full-state seasonal model samples against staged ARCO-MACDA windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.celestials.planets.mars import MARS_BODY_3D


FIELDS = {
    "surface_pressure": "psurf",
    "surface_temperature": "tsurf",
    "air_temperature": "temp",
    "eastward_wind": "uwind",
    "northward_wind": "vwind",
}


def circular_distance(values, target):
    return np.abs((np.asarray(values) - target + 180.0) % 360.0 - 180.0)


def gaussian_weights(lat, shape):
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    order = np.argsort(lat)
    if np.allclose(np.sin(np.deg2rad(np.asarray(lat)[order])), nodes, atol=1e-6):
        latitude = np.empty_like(weights)
        latitude[order] = weights
    else:
        latitude = np.cos(np.deg2rad(lat))
    reshape = (1,) * (len(shape) - 2) + (len(lat), 1)
    return np.broadcast_to(latitude.reshape(reshape), shape).copy()


def weighted_metrics(model, reference, lat):
    model = np.asarray(model, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = gaussian_weights(lat, model.shape)
    valid = np.isfinite(model) & np.isfinite(reference)
    if not valid.any():
        raise ValueError("model and reference have no finite overlap")
    x, y, w = model[valid], reference[valid], weights[valid]
    w /= w.sum()
    error = x - y
    x_mean, y_mean = np.sum(w * x), np.sum(w * y)
    x_anomaly, y_anomaly = x - x_mean, y - y_mean
    denominator = np.sqrt(np.sum(w * x_anomaly**2) * np.sum(w * y_anomaly**2))
    mean_abs_reference = np.sum(w * np.abs(y))
    return {
        "model_area_mean": float(x_mean),
        "reference_area_mean": float(y_mean),
        "bias": float(np.sum(w * error)),
        "mae": float(np.sum(w * np.abs(error))),
        "rmse": float(np.sqrt(np.sum(w * error**2))),
        "relative_mae_percent": float(
            100.0 * np.sum(w * np.abs(error)) / mean_abs_reference
        ) if mean_abs_reference else float("nan"),
        "spatial_correlation": float(
            np.sum(w * x_anomaly * y_anomaly) / denominator
        ) if denominator else float("nan"),
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_window(paths, target, half_width):
    samples = []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as opened:
            ls = float(opened.solar_longitude.isel(time=0))
            if circular_distance([ls], target)[0] <= half_width:
                sample = opened.load().assign_coords(source=("time", [str(path)]))
                samples.append(sample)
    if not samples:
        raise ValueError(f"Ls={target:g}: no model comparison samples")
    return xr.concat(samples, dim="time").mean("time", keep_attrs=True)


def load_arco_window(paths, target, half_width, mars_years):
    samples = []
    used = []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as opened:
            if int(round(float(opened.MY_Ls.isel(time=0)))) not in mars_years:
                continue
            mask = circular_distance(opened.Ls, target) <= half_width
            if bool(mask.any()):
                samples.append(opened.isel(time=np.flatnonzero(mask)).load())
                used.append(path)
    if not samples:
        raise ValueError(f"Ls={target:g}: no ARCO comparison samples")
    return xr.concat(samples, dim="time").mean("time", keep_attrs=True), used


def plot_zonal(model, reference, target, output):
    panels = (
        (model.air_temperature.mean("lon"), reference.temp.mean("lon"), "Temperature", "K"),
        (model.eastward_wind.mean("lon"), reference.uwind.mean("lon"), "Eastward wind", "m s-1"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (left, right, label, units) in enumerate(panels):
        values = (left, right, left - right)
        titles = (f"GCM {label}", f"ARCO {label}", "GCM - ARCO")
        for column, (field, title) in enumerate(zip(values, titles, strict=True)):
            image = axes[row, column].pcolormesh(
                field.lat, field.sigma, field, shading="auto",
                cmap="RdBu_r" if column == 2 else "viridis",
            )
            axes[row, column].invert_yaxis()
            axes[row, column].set(title=title, xlabel="latitude", ylabel="sigma")
            fig.colorbar(image, ax=axes[row, column], label=units)
    fig.suptitle(f"Season-matched diagnostic at Ls={target:g} degrees")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_samples", type=Path)
    parser.add_argument("arco_samples", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="45,135,225,315")
    parser.add_argument("--half-width", type=float, default=5.0)
    parser.add_argument("--mars-years", default="34,35")
    args = parser.parse_args()
    model_paths = sorted(args.model_samples.glob("sample_*.nc"))
    arco_paths = sorted(args.arco_samples.glob("macda_my*_ls*_10sol_daily_t21_l12.nc"))
    if not model_paths or not arco_paths:
        parser.error("model and ARCO sample directories must both contain NetCDF files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "diagnostic_model_reanalysis_comparison",
        "reference": "ARCO-MACDA reanalysis; not direct observational truth",
        "seasons": {},
        "limitations": [
            "Model checkpoint samples are instantaneous; ARCO fields are daily means.",
            "ARCO-MACDA retains assumptions from its parent GCM and assimilation system.",
            "Linear sigma and horizontal interpolation smooths reference extremes.",
        ],
    }
    mars_years = {int(value) for value in args.mars_years.split(",")}
    report["mars_years"] = sorted(mars_years)
    for target in [float(value) % 360.0 for value in args.targets.split(",")]:
        model = load_model_window(model_paths, target, args.half_width)
        arco, used = load_arco_window(
            arco_paths, target, args.half_width, mars_years
        )
        arco = arco.rename(lev="sigma")
        metrics = {
            name: weighted_metrics(model[name], arco[reference], model.lat)
            for name, reference in FIELDS.items()
        }
        model_wind = np.hypot(model.eastward_wind, model.northward_wind)
        arco_wind = np.hypot(arco.uwind, arco.vwind)
        metrics["wind_speed"] = weighted_metrics(model_wind, arco_wind, model.lat)
        metrics["co2_frost"] = weighted_metrics(
            model.co2_frost, arco.co2ice * MARS_BODY_3D.gravity_m_s2, model.lat
        )
        tag = f"ls{int(round(target)):03d}"
        plot_path = args.output_dir / f"zonal_{tag}.png"
        plot_zonal(model, arco, target, plot_path)
        report["seasons"][tag] = {
            "metrics": metrics,
            "arco_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in used
            ],
            "zonal_figure": str(plot_path),
        }
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
