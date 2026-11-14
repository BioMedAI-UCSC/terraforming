#!/usr/bin/env python3
"""Verify the paired winter control and plot the final model frost deficit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from audit_paper_results import field_diagnostics


def frost_deficit_map(dataset):
    air, pressure = xr.align(dataset.air_temperature, dataset.air_pressure, join="exact")
    if air.attrs.get("units") != "K" or pressure.attrs.get("units") != "Pa":
        raise ValueError("Kelvin air temperature and pascal pressure required")
    if (not np.isfinite(air).all() or not np.isfinite(pressure).all()
            or bool((pressure <= 0).any()) or bool((air <= 0).any())):
        raise ValueError("Finite positive temperature and pressure required")
    if dataset.sizes["time"] != 1:
        raise ValueError("One matched final time required")
    frost = 3182.48 / (23.3494 - np.log(np.maximum(pressure / 100., 1e-6)))
    result = (frost - air).clip(min=0).max("sigma").isel(time=0, drop=True).transpose("lat", "lon")
    result.attrs = dict(units="K", long_name="Column maximum deficit below model CO2 frost point")
    return result


def validate_controls(first, second, baseline):
    allowed = "atmospheric_condensation" if baseline == "surface-only" else "atmospheric_saturation_adjustment"
    if first["co2_forcing"].get(allowed) is not False or second["co2_forcing"].get(allowed) is not True:
        raise ValueError(f"Expected off/on pair for {allowed}")
    for key in ("grid", "forcing", "input_sha256"):
        if first[key] != second[key]:
            raise ValueError(f"Control mismatch: {key}")
    if any(c["identity"]["co2_forcing"] != c["co2_forcing"] for c in (first, second)):
        raise ValueError("Inconsistent forcing identity")
    if ({k: v for k, v in first["identity"].items() if k != "co2_forcing"}
            != {k: v for k, v in second["identity"].items() if k != "co2_forcing"}):
        raise ValueError("Control source, input or numerical identity differs")
    if ({k: v for k, v in first["co2_forcing"].items() if k != allowed}
            != {k: v for k, v in second["co2_forcing"].items() if k != allowed}):
        raise ValueError(f"Only {allowed} may change")


def analyse(root, output, baseline="surface-only"):
    if baseline not in ("surface-only", "finite-rate"):
        raise ValueError("Expected surface-only or finite-rate baseline")
    configs, datasets, maps, result = [], [], [], {}
    for name in (baseline, "clouds-on"):
        folder = root / name
        config = json.loads((folder / "configuration.json").read_text())
        report = json.loads((folder / "report.json").read_text())
        rows = json.loads((folder / "diagnostics.json").read_text())
        if report["status"] != "finite_complete" or rows[-1]["step"] != config["grid"]["n_steps"]:
            raise ValueError("Both controls must finish")
        if config["co2_forcing"]["atmospheric_condensation"] != (name == "clouds-on" or baseline == "finite-rate"):
            raise ValueError("Incorrect condensation switch")
        if baseline == "finite-rate" and config["co2_forcing"].get("atmospheric_saturation_adjustment") != (name == "clouds-on"):
            raise ValueError("Incorrect saturation-adjustment switch")
        with xr.open_dataset(folder / "final.nc") as data:
            dataset = data.load()
        deficit = frost_deficit_map(dataset)
        diagnostics = field_diagnostics(dataset, dt_seconds=config["grid"]["dt_seconds"])
        if not np.isclose(float(deficit.max()), rows[-1]["max_frost_deficit_k"], atol=1e-10, rtol=0):
            raise ValueError("Final NetCDF and recorded deficit differ")
        configs.append(config)
        datasets.append(dataset)
        maps.append(deficit)
        result[name] = dict(final=diagnostics,
            peak_sampled_frost_deficit_k=max(r["max_frost_deficit_k"] for r in rows),
            max_post_projection_relative_co2_drift=report["max_post_projection_relative_co2_drift"])
    first, second = configs
    validate_controls(first, second, baseline)
    with np.load(root / baseline / "initial.npz", allow_pickle=False) as a, np.load(
            root / "clouds-on/initial.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Initial states differ")
    a, b = xr.align(*maps, join="exact")
    for coord in ("time", "sigma", "solar_longitude"):
        if not np.array_equal(datasets[0][coord], datasets[1][coord]):
            raise ValueError(f"Final coordinate mismatch: {coord}")
    reduction = float(a.max() - b.max())
    result.update(final_worst_deficit_reduction_k=reduction,
        final_worst_deficit_reduction_percent=100 * reduction / float(a.max()),
        final_max_wind_change_ms=result["clouds-on"]["final"]["max_wind_ms"] - result[baseline]["final"]["max_wind_ms"],
        seconds=first["grid"]["n_steps"] * first["grid"]["dt_seconds"],
        limitations="Isothermal rest initialization; mechanism control, not observational accuracy. Total-pressure CO2 frost approximation. Final column maxima can occur at different levels. Extrema do not establish realistic circulation. Projected mass drift is not raw conservation.")
    output.mkdir(parents=True, exist_ok=True)
    difference = b - a
    difference.attrs = dict(units="K", long_name=f"Atmospheric-on minus {baseline} column maximum frost deficit")
    xr.Dataset({baseline.replace("-", "_"): a, "clouds_on": b, "difference": difference}, attrs=dict(
        elapsed_sols=float(datasets[0].time.values[0]), solar_longitude_deg=float(datasets[0].solar_longitude.values[0]),
        temporal_sampling="instantaneous final state", reference_role="model mechanism control; not observational error"
    )).to_netcdf(output / "frost-deficit-maps.nc")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 9), constrained_layout=True)
    axes = fig.subplots(3, 1)
    maximum = float(max(a.max(), b.max()))
    difference = b - a
    bound = float(abs(difference).max()) or 1.
    baseline_label = "Surface phase change only" if baseline == "surface-only" else "Finite-rate atmospheric phase change"
    for ax, values, title in zip(axes, (a, b, difference),
            (baseline_label, "Atmospheric and surface phase change", "Atmospheric on minus " + baseline)):
        is_difference = ax is axes[-1]
        artist = ax.pcolormesh(values.lon, values.lat, values, shading="auto",
            cmap="RdBu_r" if is_difference else "magma", vmin=-bound if is_difference else 0.,
            vmax=bound if is_difference else maximum)
        ax.set(xlabel="Longitude (degrees east)", ylabel="Latitude (degrees north)", title=title,
            xlim=(0, 360), ylim=(-90, 90))
        fig.colorbar(artist, ax=ax, label="Column maximum frost deficit (K)")
    fig.suptitle("Final winter control (~15 sols): model mechanism, not observed error")
    fig.savefig(output / "frost-deficit-maps.png", dpi=160)
    paths = [root / name / file for name in (baseline, "clouds-on")
             for file in ("configuration.json", "report.json", "diagnostics.json", "initial.npz", "final.nc")]
    result["input_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["analysis_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("audit_paper_results.py"))}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=("surface-only", "finite-rate"), default="surface-only")
    args = parser.parse_args()
    analyse(args.root, args.output, args.baseline)
