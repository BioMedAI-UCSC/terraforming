#!/usr/bin/env python3
"""Match model sampling to references; keep station and global pressure separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from benchmark_mcd import interpolate_periodic, weighted_metrics, parse_mcd_ascii

FIELDS = ("temperature", "surface_pressure", "u", "v", "co2_ice", "lowest_air_temperature", "wind_speed")


def interval_mean(times, values, start, end):
    """Trapezoidal mean with interpolated endpoints and no extrapolation."""
    if start < times[0] or end > times[-1] or end <= start:
        raise ValueError("Requested reference interval is outside model sampling")
    def at(t):
        i = np.clip(np.searchsorted(times, t)-1, 0, len(times)-2)
        fraction = (t-times[i])/(times[i+1]-times[i])
        return values[i]*(1-fraction) + values[i+1]*fraction
    inside = (times > start) & (times < end)
    x = np.r_[start, times[inside], end]
    y = np.concatenate([at(start)[None], values[inside], at(end)[None]])
    return np.trapezoid(y, x, axis=0)/(end-start)


def terrain_residual(pressure, elevation, weights):
    x, y, w = elevation.ravel(), np.log(pressure).ravel(), weights.ravel()
    if not np.isfinite(y).all() or not np.isfinite(x).all() or np.any(w <= 0):
        raise ValueError("Invalid pressure, terrain or weights")
    w = w/w.sum()
    x0, y0 = x-np.sum(w*x), y-np.sum(w*y)
    slope = np.sum(w*x0*y0)/np.sum(w*x0*x0)
    return (y0-slope*x0).reshape(pressure.shape)


def read_viking(path):
    """Read PDS VL_AVEP fixed-width records; millibar to Pa, reject gap flags."""
    rows = []
    for line in path.read_text().splitlines():
        station = line[:5].strip().strip('"')
        pressure = float(line[20:27]) * 100
        flags = line[83:87].strip()
        if station not in ("VL1", "VL2") or pressure <= 0 or any(c in flags for c in "FIC"):
            continue
        rows.append(dict(station=station, year=int(line[5:7]), ls=float(line[7:15]) % 360,
                         sol=int(line[15:20]), pressure_pa=pressure))
    if not rows:
        raise ValueError("No quality-screened Viking daily pressure records")
    return rows


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    args = p.parse_args()
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.celestials.planets.mars.gcm import radiative_forcing
    from src.celestials.planets.mars.topography import load_mola_meg, _bilinear_periodic
    period, sol = radiative_forcing().orbital_period_s, body.solar_day_seconds
    grid = np.load(args.run / "grid.npz")
    times, longitudes, values = [], [], []
    cycle, first_year_budgets = [], []
    for path in sorted((args.run / "samples").glob("sol-*.npz")):
        with np.load(path) as a:
            step_times = a["seconds"][0] + (np.arange(len(a["budgets"]))+1)*sol/300
            first_year_budgets.append(a["budgets"][step_times <= period])
            pressure = np.sum(a["fields"][:, 1] * grid["weights"], axis=(-2, -1))
            cycle.extend(dict(seconds=float(t), ls=float(ls), global_pressure_pa=float(ps))
                for t, ls, ps in zip(a["seconds"][:-1], a["ls"][:-1], pressure[:-1], strict=True))
            if a["seconds"][-1] >= period - 6*sol:
                times.append(a["seconds"][:-1])
                longitudes.append(a["ls"][:-1])
                values.append(a["fields"][:-1])
    times, ls, values = np.concatenate(times), np.concatenate(longitudes), np.concatenate(values)
    unwrapped = np.degrees(np.unwrap(np.radians(ls)))
    # The retained samples start just before the second orbit.
    while unwrapped[-1] < 360:
        unwrapped += 360
    def seasonal_time(target):
        target = target % 360 + 360
        if not unwrapped[0] <= target <= unwrapped[-1]:
            raise ValueError("Second orbit does not cover reference season")
        return float(np.interp(target, unwrapped, times))
    def dataset(field):
        return xr.Dataset({name: (("lat", "lon"), field[i].T) for i, name in enumerate(FIELDS)},
            coords=dict(lat=grid["lat"], lon=grid["lon"]))
    def instantaneous(center):
        return dataset(values[int(np.argmin(np.abs(times-center)))])
    def compare(model, reference):
        reference = interpolate_periodic(reference, model)
        metrics = {name: weighted_metrics(model[name].values, reference[name].values, model.lat.values)
                   for name in ("temperature", "surface_pressure", "u", "v", "co2_ice", "wind_speed")
                   if name in reference}
        residuals = [terrain_residual(ds.surface_pressure.values, grid["elevation_m"].T,
            grid["weights"].T) for ds in (model, reference)]
        metrics["terrain_removed_log_pressure"] = weighted_metrics(*residuals, model.lat.values)
        return metrics
    results = {}
    with xr.open_dataset(args.references / "ames.nc", decode_times=False) as ames:
        for target in (45, 135, 225, 315):
            indices = np.flatnonzero(np.abs((ames.ls.values-target+180)%360-180) <= 5)
            model_fields, reference_fields, durations = [], [], []
            for i in indices:
                center = seasonal_time(float(ames.ls[i]))
                duration = float(ames.average_duration_days[i]) * sol
                model_fields.append(interval_mean(times, values, center-duration/2, center+duration/2))
                reference_fields.append(ames.isel(ls=i).load())
                durations.append(duration)
            if not durations:
                raise ValueError("No Ames reference windows")
            model = dataset(np.average(model_fields, axis=0, weights=durations))
            reference = xr.concat(reference_fields, dim="sample").weighted(
                xr.DataArray(durations, dims="sample")).mean("sample")
            results[f"ames_ls{target}"] = dict(metrics=compare(model, reference), windows=len(durations),
                instantaneous_metrics=compare(instantaneous(seasonal_time(target)), reference),
                operator="matching five-sol intervals centered on reference Ls, duration-weighted")
    for path in sorted((args.references / "macda").glob("*.nc")):
        with xr.open_dataset(path, decode_times=False) as source:
            averages, references = [], []
            for i in range(source.sizes["time"]):
                center = seasonal_time(float(source.Ls[i]))
                averages.append(interval_mean(times, values, center-sol/2, center+sol/2))
                r = source.isel(time=i)
                references.append(xr.Dataset(dict(temperature=r.tsurf, surface_pressure=r.psurf,
                    u=r.uwind.isel(lev=-1), v=r.vwind.isel(lev=-1), co2_ice=r.co2ice*body.gravity_m_s2)).load())
            results[path.stem] = dict(metrics=compare(dataset(np.mean(averages, axis=0)),
                xr.concat(references, dim="sample").mean("sample")), windows=len(averages),
                instantaneous_metrics=compare(instantaneous(seasonal_time(float(source.Ls.mean()))),
                    xr.concat(references, dim="sample").mean("sample")),
                operator="daily model means centered on each reference daily Ls; equal day weighting",
                qualification="MACDA reanalysis; reference years differ from prescribed model dust")
    # Use each original MCD map's exact Ls and four local times, rather than a new reference.
    for path in sorted((args.references / "mcd").glob("*/benchmark.json")):
        metadata = json.loads(path.read_text())
        target = float(metadata["ls_deg"])
        center = seasonal_time(target)
        day_start = np.floor(center/sol)*sol
        local_times = metadata["local_times_hours"]
        fields = np.zeros(values.shape[1:])
        for lon_index, lon in enumerate(grid["lon"]):
            sample_times = day_start + ((np.asarray(local_times)-12-lon/15)%24)/24*sol
            for k in range(len(FIELDS)):
                for j in range(len(grid["lat"])):
                    fields[k, lon_index, j] = np.mean(np.interp(sample_times, times, values[:, k, lon_index, j]))
        reference_maps = [parse_mcd_ascii(f.read_text()) for f in sorted(path.parent.glob("mcd_ls*_lt*_z10m.txt"))]
        if len(reference_maps) != len(local_times):
            raise ValueError("MCD map/local-time count mismatch")
        ref = xr.concat(reference_maps, dim="sample").mean("sample")
        results[path.parent.name] = dict(metrics=compare(dataset(fields), ref),
            instantaneous_metrics=compare(instantaneous(center), ref),
            operator="four fixed local solar times per longitude, interpolated from model samples", ls=target,
            local_times=local_times, qualification="MCD model comparison; wind-height mismatch remains")
    observations = read_viking(args.references / "viking/vl_avep.dat")
    stations = {}
    terrain, terrain_lats, terrain_lons = load_mola_meg(args.inputs / "mola.img")
    for name, lat, lon in (("VL1", 22.48, 312.05), ("VL2", 47.97, 134.25)):
        model = xr.DataArray(values[:, [1, 5]].transpose(0, 1, 3, 2), dims=("time", "field", "lat", "lon"),
            coords=dict(time=times, lat=grid["lat"], lon=grid["lon"]))
        point = model.interp(lat=lat, lon=lon).values
        coarse_z = float(xr.DataArray(grid["elevation_m"].T, dims=("lat", "lon"),
            coords=dict(lat=grid["lat"], lon=grid["lon"])).interp(lat=lat, lon=lon))
        station_z = float(_bilinear_periodic(terrain, terrain_lats, terrain_lons, lat, lon))
        pressure = point[:, 0] * np.exp(-body.gravity_m_s2*(station_z-coarse_z)/
            (body.gas_constant_j_kg_k*point[:, 1]))
        obs, pred, seasons = [], [], []
        for lower in range(0, 360, 5):
            rows = [r for r in observations if r["station"] == name and lower <= r["ls"] < lower+5]
            if not rows:
                continue
            day_means = []
            for r in rows:
                center = seasonal_time(r["ls"])
                if center-sol/2 >= times[0] and center+sol/2 <= times[-1]:
                    day_means.append((r["pressure_pa"], interval_mean(times, pressure, center-sol/2, center+sol/2)))
            if day_means:
                a = np.mean(day_means, axis=0)
                obs.append(float(a[0])); pred.append(float(a[1])); seasons.append(lower+2.5)
        obs, pred = np.asarray(obs), np.asarray(pred)
        stations[name] = dict(latitude=lat, longitude_east=lon, station_mola_elevation_m=station_z,
            model_elevation_m=coarse_z, ls=seasons, observed_pressure_pa=obs.tolist(),
            model_pressure_pa=pred.tolist(), pressure_rmse_pa=float(np.sqrt(np.mean((pred-obs)**2))),
            anomaly_correlation=float(np.corrcoef(obs, pred)[0, 1]),
            observed_amplitude_pa=float(np.ptp(obs)), model_amplitude_pa=float(np.ptp(pred)),
            observed_minimum_ls=seasons[int(np.argmin(obs))], model_minimum_ls=seasons[int(np.argmin(pred))],
            observed_maximum_ls=seasons[int(np.argmax(obs))], model_maximum_ls=seasons[int(np.argmax(pred))],
            operator="quality-screened PDS daily means; model daily means at each observed Ls, then 5-degree bins",
            limitation="Prescribed dust is not Viking-year dust; hydrostatic height correction uses lowest-layer temperature. Seasonal climatology comparison, not contemporaneous weather skill.")
    budgets = np.concatenate(first_year_budgets)
    initial_mass = budgets[0, 1]
    correction = budgets[:, 3]/initial_mass
    first_year = dict(steps=len(budgets), duration_seconds=len(budgets)*sol/300,
        mean_absolute_correction_fraction=float(np.mean(np.abs(correction))),
        max_absolute_correction_fraction=float(np.max(np.abs(correction))),
        mean_negative_frost_repair_fraction=float(np.mean(budgets[:, 4]/initial_mass)),
        max_negative_frost_repair_fraction=float(np.max(budgets[:, 4]/initial_mass)),
        cumulative_signed_correction_fraction=float(np.sum(correction)),
        cumulative_absolute_correction_fraction=float(np.sum(np.abs(correction))),
        maximum_relative_mass_drift=float(np.max(np.abs(budgets[:, 2]/initial_mass-1))),
        final_relative_mass_drift=float(budgets[-1, 2]/initial_mass-1))
    report = clean(dict(comparisons=results, viking=stations, global_pressure_cycle=cycle,
        first_orbital_year=first_year,
        limitations="Second-orbit cold-start evaluation; no equilibrium certification. Temporal integrals use hourly trapezoidal quadrature. Pressure terrain removal is independent weighted log-pressure regression on elevation."))
    (args.run / "reference-analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
