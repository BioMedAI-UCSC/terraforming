#!/usr/bin/env python3
"""Evaluate a completed MOLA physical baseline against staged ARCO-MACDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.celestials.planets.mars import MARS_BODY_3D


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights(lat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    nodes, gaussian = np.polynomial.legendre.leggauss(len(lat))
    order = np.argsort(lat)
    if np.allclose(np.sin(np.deg2rad(lat[order])), nodes, atol=1e-6):
        latitude = np.empty_like(gaussian)
        latitude[order] = gaussian
    else:
        latitude = np.cos(np.deg2rad(lat))
    result = np.broadcast_to(latitude[:, None], shape).astype(np.float64).copy()
    return result / result.sum()


def _metrics(model: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(model) & np.isfinite(reference) & np.isfinite(weights)
    x, y, w = model[valid], reference[valid], weights[valid]
    w = w / w.sum()
    error = x - y
    x_mean = float(np.sum(w * x))
    y_mean = float(np.sum(w * y))
    rmse = float(np.sqrt(np.sum(w * error**2)))
    mae = float(np.sum(w * np.abs(error)))
    y_std = float(np.sqrt(np.sum(w * (y - y_mean) ** 2)))
    mean_abs_y = float(np.sum(w * np.abs(y)))
    covariance = float(np.sum(w * (x - x_mean) * (y - y_mean)))
    x_std = float(np.sqrt(np.sum(w * (x - x_mean) ** 2)))
    return {
        "model_area_mean": x_mean,
        "reference_area_mean": y_mean,
        "bias": float(np.sum(w * error)),
        "mae": mae,
        "rmse": rmse,
        "relative_mae_percent_of_reference_mean_absolute": (
            100.0 * mae / mean_abs_y if mean_abs_y else float("nan")
        ),
        "normalized_rmse_percent_of_reference_spatial_std": (
            100.0 * rmse / y_std if y_std else float("nan")
        ),
        "spatial_correlation": (
            covariance / (x_std * y_std) if x_std and y_std else float("nan")
        ),
    }


def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mean = float(np.sum(weights * values))
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "area_weighted_mean": mean,
        "area_weighted_std": float(np.sqrt(np.sum(weights * (values - mean) ** 2))),
    }


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x_mean = np.sum(weights * x)
    y_mean = np.sum(weights * y)
    covariance = np.sum(weights * (x - x_mean) * (y - y_mean))
    denominator = math.sqrt(
        np.sum(weights * (x - x_mean) ** 2)
        * np.sum(weights * (y - y_mean) ** 2)
    )
    return float(covariance / denominator)


def _weighted_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x_mean = np.sum(weights * x)
    y_mean = np.sum(weights * y)
    return float(
        np.sum(weights * (x - x_mean) * (y - y_mean))
        / np.sum(weights * (x - x_mean) ** 2)
    )


def _circular_distance(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - target + 180.0) % 360.0 - 180.0)


def _read_diagnostics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            trailing = row.pop(None, None)
            if trailing:
                if len(trailing) != 2:
                    raise ValueError(
                        f"unexpected trailing diagnostic columns in {path}: {trailing}"
                    )
                row["mean_dust_visible_optical_depth"] = trailing[0]
                row["mean_dust_longwave_optical_depth"] = trailing[1]
            rows.append(
                {
                    key: (float(value) if value not in (None, "") else float("nan"))
                    for key, value in row.items()
                }
            )
        return rows


def _write_normalized_diagnostics(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["step"] = int(serialized["step"])
            writer.writerow(serialized)


def _plot_comparison(
    model: xr.Dataset,
    reference: dict[str, xr.DataArray],
    output: Path,
) -> None:
    fields = [
        ("surface_pressure", "Surface pressure", "Pa"),
        ("temperature", "Surface temperature", "K"),
        ("wind_speed", "Near-surface wind speed", "m s-1"),
        ("co2_ice", "Surface CO2 ice", "Pa-equivalent"),
    ]
    fig, axes = plt.subplots(len(fields), 3, figsize=(15, 13), constrained_layout=True)
    for row, (name, label, units) in enumerate(fields):
        model_values = np.asarray(model[name])
        reference_values = np.asarray(reference[name])
        panels = (model_values, reference_values, model_values - reference_values)
        titles = (f"Dinosaur {label}", f"ARCO-MACDA {label}", "Dinosaur - ARCO-MACDA")
        for column, (values, title) in enumerate(zip(panels, titles, strict=True)):
            cmap = "RdBu_r" if column == 2 else "viridis"
            image = axes[row, column].pcolormesh(
                model.lon, model.lat, values, shading="auto", cmap=cmap
            )
            axes[row, column].set(title=title, xlabel="longitude (deg E)", ylabel="latitude (deg N)")
            fig.colorbar(image, ax=axes[row, column], label=units, shrink=0.8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_mola_pressure(model: xr.Dataset, weights: np.ndarray, output: Path) -> None:
    elevation = np.asarray(model.elevation)
    pressure = np.asarray(model.surface_pressure)
    slope = _weighted_slope(elevation / 1000.0, pressure, weights)
    intercept = np.sum(weights * pressure) - slope * np.sum(weights * elevation / 1000.0)
    line_x = np.linspace(elevation.min() / 1000.0, elevation.max() / 1000.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image = axes[0].pcolormesh(model.lon, model.lat, elevation, shading="auto", cmap="terrain")
    axes[0].set(title="MOLA elevation on T21 grid", xlabel="longitude (deg E)", ylabel="latitude (deg N)")
    fig.colorbar(image, ax=axes[0], label="m")
    axes[1].scatter(elevation.ravel() / 1000.0, pressure.ravel(), s=5, alpha=0.35)
    axes[1].plot(line_x, intercept + slope * line_x, color="black", linewidth=2)
    axes[1].set(
        title="Final surface pressure versus MOLA elevation",
        xlabel="elevation (km)",
        ylabel="surface pressure (Pa)",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("arco", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with xr.open_dataset(args.model) as opened:
        model = opened.load()
    with xr.open_dataset(args.arco, decode_times=False) as opened:
        arco = opened.load()
    rows = _read_diagnostics(args.diagnostics)
    if not rows or rows[0]["step"] != 0:
        raise ValueError("diagnostics must include the exact step-zero inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_diagnostics = args.output_dir / "checkpoint_diagnostics_normalized.csv"
    _write_normalized_diagnostics(rows, normalized_diagnostics)

    model_ls = float(model.attrs["solar_longitude_deg"])
    arco_index = int(np.argmin(_circular_distance(np.asarray(arco.Ls), model_ls)))
    reference_day = arco.isel(time=arco_index)
    wind_sigma = float(model.attrs["wind_level_sigma"])
    lev_index = int(np.argmin(np.abs(np.asarray(reference_day.lev) - wind_sigma)))
    atmospheric = reference_day.isel(lev=lev_index)
    reference = {
        "surface_pressure": reference_day.psurf,
        "temperature": reference_day.tsurf,
        "u": atmospheric.uwind,
        "v": atmospheric.vwind,
        "wind_speed": np.hypot(atmospheric.uwind, atmospheric.vwind),
        "co2_ice": reference_day.co2ice * MARS_BODY_3D.gravity_m_s2,
    }
    reference["co2_ice"].attrs["units"] = "Pa-equivalent"

    shape = tuple(model.surface_pressure.shape)
    weights = _weights(np.asarray(model.lat), shape)
    metrics = {
        name: _metrics(np.asarray(model[name]), np.asarray(reference[name]), weights)
        for name in ("surface_pressure", "temperature", "u", "v", "wind_speed", "co2_ice")
    }

    elevation = np.asarray(model.elevation)
    pressure = np.asarray(model.surface_pressure)
    log_pressure_slope = _weighted_slope(elevation, np.log(pressure), weights)
    mola = _weighted_summary(elevation, weights)
    mola.update(
        surface_pressure_elevation_correlation=_weighted_correlation(
            elevation, pressure, weights
        ),
        surface_pressure_slope_pa_per_km=_weighted_slope(
            elevation / 1000.0, pressure, weights
        ),
        fitted_log_pressure_scale_height_m=(
            -1.0 / log_pressure_slope if log_pressure_slope < 0.0 else float("nan")
        ),
        representation="MOLA elevation regridded and spectrally represented on the T21 Gaussian grid",
    )

    masses = np.asarray([row["atmosphere_plus_surface_reservoir_mass_kg"] for row in rows])
    initial_mass = float(masses[0])
    drift = masses - initial_mass
    final = rows[-1]
    completed_sols = float(final["elapsed_sols"])
    core_diagnostic_fields = (
        "elapsed_sols",
        "mean_surface_pressure_pa",
        "mean_co2_ice_pa",
        "atmosphere_plus_surface_reservoir_mass_kg",
        "mean_surface_temperature_k",
        "min_surface_temperature_k",
        "max_surface_temperature_k",
        "mean_deep_soil_temperature_k",
        "dry_atmospheric_mass_kg_m2_sr",
        "atmospheric_energy_excluding_surface_geopotential_j_m2_sr",
        "dry_atmospheric_aam_kg_m_s_sr",
    )
    finite = bool(
        all(np.isfinite(np.asarray(variable)).all() for variable in model.data_vars.values())
        and all(
            np.isfinite([row[name] for name in core_diagnostic_fields]).all()
            for row in rows
        )
    )
    conservation = {
        "initial_total_co2_mass_kg": initial_mass,
        "final_total_co2_mass_kg": float(masses[-1]),
        "final_drift_kg": float(drift[-1]),
        "final_relative_drift": float(drift[-1] / initial_mass),
        "final_drift_percent": float(100.0 * drift[-1] / initial_mass),
        "maximum_absolute_checkpoint_drift_kg": float(np.max(np.abs(drift))),
        "maximum_absolute_checkpoint_relative_drift": float(
            np.max(np.abs(drift)) / initial_mass
        ),
        "checkpoint_count": len(rows),
    }

    comparison_plot = args.output_dir / "arco_macda_comparison.png"
    mola_plot = args.output_dir / "mola_topography_pressure.png"
    _plot_comparison(model, reference, comparison_plot)
    _plot_mola_pressure(model, weights, mola_plot)

    report = {
        "status": (
            "stable_completed_requested_duration"
            if finite and completed_sols >= 667.9
            else "incomplete_or_nonfinite"
        ),
        "completed_sols": completed_sols,
        "all_exported_and_checkpoint_values_finite": finite,
        "model_solar_longitude_deg": model_ls,
        "arco_reference": {
            "mars_year": float(reference_day.MY_Ls),
            "daily_mean_solar_longitude_deg": float(reference_day.Ls),
            "absolute_season_offset_deg": float(
                _circular_distance(np.asarray([float(reference_day.Ls)]), model_ls)[0]
            ),
            "wind_sigma": float(reference_day.lev.isel(lev=lev_index)),
            "model_wind_sigma": wind_sigma,
        },
        "metrics": metrics,
        "mola_topography": mola,
        "co2_mass_conservation": conservation,
        "final_state_ranges": {
            name: {
                "minimum": float(np.min(np.asarray(model[name]))),
                "maximum": float(np.max(np.asarray(model[name]))),
            }
            for name in ("surface_pressure", "temperature", "wind_speed", "co2_ice")
        },
        "artifacts": {
            "model_netcdf": str(args.model),
            "model_sha256": _sha256(args.model),
            "arco_netcdf": str(args.arco),
            "arco_sha256": _sha256(args.arco),
            "diagnostics_csv": str(args.diagnostics),
            "diagnostics_sha256": _sha256(args.diagnostics),
            "normalized_diagnostics_csv": str(normalized_diagnostics),
            "normalized_diagnostics_sha256": _sha256(normalized_diagnostics),
            "comparison_plot": str(comparison_plot),
            "mola_plot": str(mola_plot),
        },
        "metric_definitions": {
            "weighting": "Gaussian spherical quadrature on the T21 latitude grid",
            "relative_mae": "100 * area-weighted MAE / area-weighted mean absolute reference",
            "normalized_rmse": "100 * area-weighted RMSE / reference spatial standard deviation",
        },
        "limitations": [
            "ARCO-MACDA is a data-assimilating reanalysis and not direct observational truth.",
            "The model field is one instantaneous final state under daily-mean solar forcing; ARCO-MACDA is a daily mean.",
            "A completed first Mars year demonstrates numerical stability, not climatological equilibration.",
            "ARCO-MACDA was linearly regridded from 35 sigma levels and 36x72 cells to T21/L12, which smooths extremes.",
            "The MY24 sample is a pipeline and diagnostic comparison; it is not a held-out test score.",
        ],
    }
    output = args.output_dir / "baseline_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
