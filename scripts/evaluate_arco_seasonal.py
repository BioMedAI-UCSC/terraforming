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
from src.celestials.planets.mars.mcd import interpolate_periodic


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
    lat = np.asarray(lat, dtype=float)
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
    if isinstance(model, xr.DataArray) and isinstance(reference, xr.DataArray):
        model, reference = xr.align(model, reference.transpose(*model.dims), join="exact")
    model = np.asarray(model, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if model.shape != reference.shape or model.shape[-2] != len(lat):
        raise ValueError("Model and reference must have matching labelled grids")
    weights = gaussian_weights(lat, model.shape)
    valid = np.isfinite(model) & np.isfinite(reference)
    if not valid.all():
        raise ValueError("Incomplete model/reference coverage; refusing to drop comparison cells")
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
        ) if mean_abs_reference else None,
        "spatial_correlation": float(
            np.sum(w * x_anomaly * y_anomaly) / denominator
        ) if denominator else None,
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_window(paths, target, half_width):
    samples, used = [], []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as opened:
            mask = circular_distance(opened.solar_longitude, target) <= half_width
            if bool(mask.any()):
                sample = opened.isel(time=np.flatnonzero(mask)).load()
                sample["wind_speed"] = np.hypot(sample.eastward_wind, sample.northward_wind)
                samples.append(sample)
                used.append(path)
    if not samples:
        raise ValueError(f"Ls={target:g}: no model comparison samples")
    combined = xr.concat(samples, dim="time", join="exact")
    if not combined.indexes["time"].is_unique:
        raise ValueError("Repeated model times would double-weight seasonal maps")
    return combined.mean("time", keep_attrs=True, skipna=False), used


def load_arco_window(paths, target, half_width, mars_years):
    samples = []
    used = []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as opened:
            mask = ((circular_distance(opened.Ls, target) <= half_width)
                    & np.isin(np.rint(opened.MY_Ls), list(mars_years)))
            if bool(mask.any()):
                sample = opened.isel(time=np.flatnonzero(mask)).load()
                units = dict(temp="K", tsurf="K", psurf="Pa", uwind="m s-1", vwind="m s-1", co2ice="kg m-2")
                for name, expected in units.items():
                    if sample[name].attrs.get("units") != expected:
                        raise ValueError(f"ARCO {name}: expected units {expected}")
                sample["wind_speed"] = np.hypot(sample.uwind, sample.vwind)
                samples.append(sample)
                used.append(path)
    if not samples:
        raise ValueError(f"Ls={target:g}: no ARCO comparison samples")
    combined = xr.concat(samples, dim="time", join="exact")
    if not combined.indexes["time"].is_unique:
        raise ValueError("Repeated reference times would double-weight seasonal maps")
    return combined.mean("time", keep_attrs=True, skipna=False), used


def align_reference(reference, model):
    reference = reference.rename(lev="sigma")
    # Staged float32 sigma centres can differ from the model only by roundoff.
    if reference.sigma.shape == model.sigma.shape and np.allclose(reference.sigma, model.sigma, rtol=0, atol=1e-7):
        reference = reference.assign_coords(sigma=model.sigma)
    else:
        reference = reference.sortby("sigma").interp(sigma=model.sigma)
    return interpolate_periodic(reference, model)


def wind_at_height(temperature, eastward, northward, gas_constant, height_m=1000.):
    """Interpolate seasonal-mean profiles to one hydrostatic height above ground."""
    from src.framework.physics.co2_clouds import geopotential_matrix
    from src.framework.gcm._dinosaur import jax
    temperature, eastward, northward = xr.align(temperature, eastward, northward, join="exact")
    temperature = temperature.transpose("sigma", "lat", "lon")
    air = np.asarray(temperature)
    if not np.isfinite(air).all() or np.any(air <= 0):
        raise ValueError("Finite positive temperatures required for hydrostatic height")
    sigma = np.asarray(temperature.sigma)
    if not np.all((sigma > 0) & (sigma < 1)) or np.any(np.diff(sigma) <= 0):
        raise ValueError("Increasing sigma centres strictly inside (0, 1) required")
    with jax.enable_x64(True):
        geometry = np.asarray(geopotential_matrix(sigma, gas_constant))
    height = np.einsum("ij,jxy->ixy", geometry, air) / MARS_BODY_3D.gravity_m_s2
    if np.any(height_m < height[-1]) or np.any(height_m > height[0]):
        raise ValueError("Requested height is outside resolved layers; no wind extrapolation")
    result = {}
    for name, field in (("eastward_wind", eastward), ("northward_wind", northward)):
        values = np.asarray(field.transpose(*temperature.dims))
        if not np.isfinite(values).all():
            raise ValueError("Finite winds required for height comparison")
        output = np.empty(air.shape[1:])
        for i, j in np.ndindex(output.shape):
            output[i, j] = np.interp(height_m, height[::-1, i, j], values[::-1, i, j])
        result[name] = (("lat", "lon"), output, {"units": "m s-1"})
    ds = xr.Dataset(result, coords=dict(lat=temperature.lat, lon=temperature.lon),
        attrs=dict(height_m_above_ground=height_m, gas_constant_j_kg_k=gas_constant,
                   sampling="Hydrostatic interpolation of seasonal-mean profiles; magnitude of mean vector, not mean instantaneous speed"))
    ds["wind_speed"] = np.hypot(ds.eastward_wind, ds.northward_wind)
    ds.wind_speed.attrs["units"] = "m s-1"
    return ds


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
            "Reference speed is the averaged magnitude of daily-mean wind vectors; subdaily reference speed variability is unavailable.",
            "ARCO-MACDA retains assumptions from its parent GCM and assimilation system.",
            "Linear sigma and horizontal interpolation smooths reference extremes.",
        ],
    }
    mars_years = {int(value) for value in args.mars_years.split(",")}
    report["mars_years"] = sorted(mars_years)
    for target in [float(value) % 360.0 for value in args.targets.split(",")]:
        model, model_used = load_model_window(model_paths, target, args.half_width)
        arco, used = load_arco_window(
            arco_paths, target, args.half_width, mars_years
        )
        arco = align_reference(arco, model)
        metrics = {
            name: weighted_metrics(model[name], arco[reference], model.lat)
            for name, reference in FIELDS.items()
        }
        metrics["wind_speed"] = weighted_metrics(model.wind_speed, arco.wind_speed, model.lat)
        metrics["co2_frost"] = weighted_metrics(
            model.co2_frost, arco.co2ice * MARS_BODY_3D.gravity_m_s2, model.lat
        )
        tag = f"ls{int(round(target)):03d}"
        model_height = wind_at_height(model.air_temperature, model.eastward_wind, model.northward_wind,
                                      MARS_BODY_3D.gas_constant_j_kg_k)
        # ARCO's native temperature metadata specifies Rspec=191.182 J/(kg K).
        arco_height = wind_at_height(arco.temp, arco.uwind, arco.vwind, 191.182)
        for field in ("eastward_wind", "northward_wind", "wind_speed"):
            metrics[field + "_1000m_mean_profile"] = weighted_metrics(model_height[field], arco_height[field], model.lat)
        model_height.to_netcdf(args.output_dir / f"model_wind_1000m_{tag}.nc")
        arco_height.to_netcdf(args.output_dir / f"arco_wind_1000m_{tag}.nc")
        plot_path = args.output_dir / f"zonal_{tag}.png"
        plot_zonal(model, arco, target, plot_path)
        report["seasons"][tag] = {
            "metrics": metrics,
            "fixed_height_wind": dict(height_m_above_ground=1000., model_gas_constant_j_kg_k=MARS_BODY_3D.gas_constant_j_kg_k,
                reference_gas_constant_j_kg_k=191.182, sampling=model_height.attrs["sampling"]),
            "model_files": [{"path": str(path), "sha256": _sha256(path)} for path in model_used],
            "arco_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in used
            ],
            "zonal_figure": str(plot_path),
        }
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
