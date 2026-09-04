"""FastAPI visualisation server for tform.

Started by ``tform serve``.  Runs simulations in a thread pool, streams every
physics step to the browser via Server-Sent Events, and serves the pre-built
React UI from cli/static/.

Callback chain
--------------
TimeController.run(callback=step_cb)
    └── step_cb(planet, elapsed)          per integration step
         └── _runs[id]["data"].append()   thread-safe via CPython GIL
              └── SSE generator           polls list with cursor
                   └── EventSource        browser updates Recharts in real-time

For intervention runs the InterventionController provides a per-year callback
(InterventionSnapshot) which is the natural chart granularity.  For sol/year
runs the per-step callback is throttled to at most MAX_CHART_POINTS points.
"""

from __future__ import annotations

import asyncio
import csv as _csv
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="tform visualizer", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)
_runs: dict[str, dict[str, Any]] = {}
_mcd_ascii_cache: dict[tuple[float, float, int, float], tuple[str, str]] = {}

MAX_CHART_POINTS = 2000  # throttle cap for sol/year runs
_STATIC_DIR  = Path(__file__).parent / "static"
_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "server"
_BENCHMARK_DATA_DIR = Path(__file__).parent.parent / "outputs" / "data"


# ── Request model ──────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    preset: str = "current-mars"
    exp_type: str = "intervention"
    years: int = 100
    sols: float = 1.0
    accuracy: str = "fast"
    dt: float = 3600.0
    lat: float | None = None
    lon: float | None = None
    elevation: float | None = None
    ls: float | None = None
    surface_temp: float | None = None
    surface_pressure: float | None = None
    albedo: float | None = None
    greenhouse_factor: float | None = None
    ice_mass: float | None = None
    inject: dict[str, float] = {}
    label: str | None = None
    # gcm ('gcm3d maps') options
    scale: str = "fast"      # runtime resolution preset (fast/balanced/high/ultra)
    snapshots: int = 5       # 3-D map snapshots along an intervention timeline
    diurnal: bool = False    # moving day/night terminator (else daily-mean insolation)
    compare_mcd: bool = False
    mcd_local_time: float | None = None  # None = 12-sample diurnal mean
    mcd_dust: int = 1


class MCDBenchmarkRequest(BaseModel):
    ls: float | None = None
    local_time: float | None = None
    dust: int = 1
    auto_match: bool = True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _v(t) -> float:
    return float(t.item())


def _default_label(req: RunRequest) -> str:
    if req.exp_type == "intervention":
        compounds = "+".join(req.inject.keys()) or "baseline"
        return f"{req.years}yr {compounds}"
    if req.exp_type == "year":
        return f"{req.preset} {req.years}yr"
    if req.exp_type == "sol":
        return f"{req.preset} {req.sols}sol"
    return f"{req.preset} {req.exp_type}"


def _save_run_csv(run_id: str, run: dict) -> None:
    if not run["data"]:
        return
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = run["label"][:40].replace("/", "-").replace(" ", "_")
    path = _OUTPUTS_DIR / f"{slug}_{run_id}.csv"
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(run["data"][0].keys()))
        writer.writeheader()
        writer.writerows(run["data"])
    run["csv_path"] = str(path)


# ── Simulation thread ──────────────────────────────────────────────────────────

def _extract_maps_fields(fields) -> dict:
    """Adapt a gcm3d ``MarsMapFields`` into the browser field-grid contract.

    Returns lon/lat axes plus 2-D maps (row = lat, col = lon) each with its own
    min/max/units, matching the ``RunFields`` shape the ``FieldMap`` component
    consumes. gcm3d already stores every field as ``(n_lat, n_lon)``.
    """
    import numpy as np

    def _map(arr, label, units):
        a = np.asarray(arr, dtype=float)
        return {"label": label, "units": units,
                "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                "data": np.round(a, 3).tolist()}

    maps = {
        "surface_temperature": _map(fields.temperature_k, "Surface temperature", "K"),
        "surface_pressure": _map(fields.surface_pressure_pa, "Surface pressure", "Pa"),
        "surface_zonal_wind": _map(fields.u_ms, "Surface zonal wind", "m/s"),
        "surface_meridional_wind": _map(
            fields.v_ms, "Surface meridional wind", "m/s"
        ),
        "surface_wind_speed": _map(fields.wind_speed_ms, "Surface wind speed", "m/s"),
        "elevation": _map(fields.elevation_m, "MOLA elevation", "m"),
    }
    if getattr(fields, "co2_ice_pa", None) is not None:
        maps["co2_ice"] = _map(fields.co2_ice_pa, "CO2 surface frost", "Pa-equiv")

    return {
        "lon": np.round(np.asarray(fields.lon_deg), 2).tolist(),
        "lat": np.round(np.asarray(fields.lat_deg), 2).tolist(),
        "sigma": [],
        "maps": maps,
        "sections": {},
        "metadata": {
            "fidelity": getattr(fields, "physics", "unspecified diagnostic physics"),
            "duration_sols": float(getattr(fields, "duration_sols", 0.0)),
            "is_transient": bool(getattr(fields, "is_transient", True)),
            "truncation": getattr(fields, "truncation", "unknown"),
            "n_layers": int(getattr(fields, "n_layers", 0)),
        },
    }


def _state_map_snapshot(state, coords, specs, body) -> dict:
    """Convert an in-progress column state to the existing global map contract."""
    import numpy as np
    from src.framework.gcm._dinosaur import scales, spherical_harmonic
    from src.framework.gcm.dynamics import reference_temperature

    dyn, grid = state.dynamics, coords.horizontal
    ps = np.asarray(specs.dimensionalize(
        np.exp(grid.to_nodal(dyn.log_surface_pressure)), scales.units.pascal
    ).magnitude)[0].T
    ref = np.asarray(reference_temperature(coords, body))[:, None, None]
    air = np.asarray(specs.dimensionalize(
        grid.to_nodal(dyn.temperature_variation) + ref, scales.units.kelvin
    ).magnitude)[-1].T
    surface = np.asarray(specs.dimensionalize(
        state.surface_temperature, scales.units.kelvin
    ).magnitude)[0].T
    u_nd, v_nd = spherical_harmonic.vor_div_to_uv_nodal(
        grid, dyn.vorticity, dyn.divergence
    )
    velocity_unit = scales.units.meter / scales.units.second
    u = np.asarray(specs.dimensionalize(u_nd, velocity_unit).magnitude)[-1].T
    v = np.asarray(specs.dimensionalize(v_nd, velocity_unit).magnitude)[-1].T
    ice = np.asarray(specs.dimensionalize(
        state.co2_ice, scales.units.pascal
    ).magnitude)[0].T
    time_scale = float(specs.nondimensionalize(1.0 * scales.units.second))
    elapsed_s = float(dyn.sim_time) / time_scale
    return {
        "elapsed_seconds": elapsed_s,
        "arrays": {
            "surface_temperature": surface, "near_surface_air_temperature": air,
            "surface_pressure": ps, "surface_zonal_wind": u,
            "surface_meridional_wind": v, "surface_wind_speed": np.hypot(u, v),
            "co2_ice": ice,
        },
    }


def _assemble_fixed_local_time_maps(raw_snapshots: list[dict], fields: dict,
                                    rotation_period_s: float) -> dict:
    """Assemble global fixed-LT maps from instantaneous maps over the last sol."""
    import numpy as np

    if not raw_snapshots:
        return {"local_times_hours": [], "snapshots": {}}
    final_s = raw_snapshots[-1]["elapsed_seconds"]
    raw = [s for s in raw_snapshots
           if s["elapsed_seconds"] >= final_s - 1.02 * rotation_period_s]
    elapsed = np.asarray([s["elapsed_seconds"] for s in raw])
    lon = np.asarray(fields["lon"])
    prime_lmst = np.mod(12.0 + 24.0 * elapsed / rotation_period_s, 24.0)
    local_time = np.mod(prime_lmst[:, None] + lon[None, :] / 15.0, 24.0)
    labels = {
        "surface_temperature": ("Surface temperature", "K"),
        "near_surface_air_temperature": ("Near-surface air temperature", "K"),
        "surface_pressure": ("Surface pressure", "Pa"),
        "surface_zonal_wind": ("Surface zonal wind", "m/s"),
        "surface_meridional_wind": ("Surface meridional wind", "m/s"),
        "surface_wind_speed": ("Surface wind speed", "m/s"),
        "co2_ice": ("CO2 surface frost", "Pa-equiv"),
    }
    def grid(array, label, units):
        return {"label": label, "units": units, "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)), "data": np.round(array, 3).tolist()}

    snapshots = {}
    for target in range(0, 24, 3):
        distance = np.abs((local_time - target + 12.0) % 24.0 - 12.0)
        indices = np.argmin(distance, axis=0)
        maps = {}
        for name, (label, units) in labels.items():
            stack = np.stack([s["arrays"][name] for s in raw])
            assembled = np.stack([stack[indices[j], :, j] for j in range(lon.size)], axis=1)
            maps[name] = grid(assembled, f"{label} · LT {target:02d}:00", units)
        maps["elevation"] = fields["maps"]["elevation"]
        snapshots[str(target)] = {"maps": maps}
    return {
        "local_times_hours": list(range(0, 24, 3)), "snapshots": snapshots,
        "metadata": {"method": "fixed-local-time composites from the final simulated sol",
                     "samples_in_final_sol": len(raw)},
    }


def _fields_netcdf_bytes(fields: dict) -> bytes:
    """Serialize headline and fixed-local-time grids into one NetCDF artifact."""
    import numpy as np
    import xarray as xr

    coords = {"lat": np.asarray(fields["lat"]), "lon": np.asarray(fields["lon"])}
    variables = {}
    for name, grid in fields["maps"].items():
        variables[name] = (("lat", "lon"), np.asarray(grid["data"]),
                           {"units": grid["units"], "long_name": grid["label"]})
    diurnal = fields.get("diurnal")
    if diurnal and diurnal["local_times_hours"]:
        hours = diurnal["local_times_hours"]
        coords["local_time"] = np.asarray(hours, dtype=float)
        first = diurnal["snapshots"][str(hours[0])]["maps"]
        for name, grid in first.items():
            stack = np.asarray([
                diurnal["snapshots"][str(hour)]["maps"][name]["data"]
                for hour in hours
            ])
            variables[f"diurnal_{name}"] = (
                ("local_time", "lat", "lon"), stack,
                {"units": grid["units"], "long_name": grid["label"].split(" · LT")[0]},
            )
    ds = xr.Dataset(variables, coords=coords, attrs=fields.get("metadata", {}))
    if "local_time" in ds.coords:
        ds.local_time.attrs.update(units="hour", long_name="local mean solar time")
    return ds.to_netcdf()


def _imported_netcdf_fields(payload: bytes, filename: str) -> dict:
    """Load a previously exported DMGCM NetCDF into the browser map contract."""
    import tempfile
    import numpy as np
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".nc") as tmp:
        tmp.write(payload); tmp.flush()
        ds = xr.open_dataset(tmp.name).load()
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    ds = ds.rename(rename)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("DMGCM NetCDF requires lat and lon coordinates")
    aliases = {
        "surface_temperature": ("surface_temperature", "temperature"),
        "near_surface_air_temperature": ("near_surface_air_temperature",),
        "surface_pressure": ("surface_pressure", "ps"),
        "surface_zonal_wind": ("surface_zonal_wind", "u"),
        "surface_meridional_wind": ("surface_meridional_wind", "v"),
        "surface_wind_speed": ("surface_wind_speed", "wind_speed"),
        "co2_ice": ("co2_ice",), "elevation": ("elevation", "topography"),
    }
    defaults = {
        "surface_temperature": ("Surface temperature", "K"),
        "near_surface_air_temperature": ("Near-surface air temperature", "K"),
        "surface_pressure": ("Surface pressure", "Pa"),
        "surface_zonal_wind": ("Surface zonal wind", "m/s"),
        "surface_meridional_wind": ("Surface meridional wind", "m/s"),
        "surface_wind_speed": ("Surface wind speed", "m/s"),
        "co2_ice": ("CO2 surface frost", "Pa-equiv"),
        "elevation": ("MOLA elevation", "m"),
    }

    def read_grid(da, name):
        while da.ndim > 2:
            da = da.isel({da.dims[0]: -1})
        values = np.asarray(da.transpose("lat", "lon"), dtype=float)
        label, units = defaults[name]
        return {"label": str(da.attrs.get("long_name", label)),
                "units": str(da.attrs.get("units", units)),
                "min": float(np.nanmin(values)), "max": float(np.nanmax(values)),
                "data": np.round(values, 3).tolist()}

    maps = {}
    for name, candidates in aliases.items():
        variable = next((candidate for candidate in candidates if candidate in ds), None)
        if variable:
            maps[name] = read_grid(ds[variable], name)
    if not {"surface_temperature", "surface_pressure"} <= set(maps):
        raise ValueError("NetCDF needs DMGCM temperature and surface-pressure fields")
    fields = {
        "lon": np.asarray(ds.lon, dtype=float).tolist(),
        "lat": np.asarray(ds.lat, dtype=float).tolist(), "sigma": [],
        "maps": maps, "sections": {},
        "metadata": {"fidelity": str(ds.attrs.get("fidelity", "imported DMGCM output")),
                     "duration_sols": float(ds.attrs.get("duration_sols", 0.0)),
                     "is_transient": bool(ds.attrs.get("is_transient", True)),
                     "truncation": str(ds.attrs.get("truncation", "imported")),
                     "n_layers": int(ds.attrs.get("n_layers", 0) or 0),
                     "source_filename": filename},
    }
    if "local_time" in ds.coords:
        hours = [float(value) for value in np.asarray(ds.local_time)]
        snapshots = {}
        for index, hour in enumerate(hours):
            snapshot_maps = {}
            for name in aliases:
                variable = f"diurnal_{name}"
                if variable in ds:
                    snapshot_maps[name] = read_grid(ds[variable].isel(local_time=index), name)
            if snapshot_maps:
                snapshots[str(int(hour) if hour.is_integer() else hour)] = {"maps": snapshot_maps}
        fields["diurnal"] = {"local_times_hours": hours, "snapshots": snapshots,
                             "metadata": {"method": "loaded from NetCDF local_time dimension",
                                          "samples_in_final_sol": len(hours)}}
    return fields


def _matched_mcd_comparison(fields, ls_deg: float, local_time: float | None,
                            dust: int = 1) -> dict:
    """Fetch MCD with the requested season/time/height and align it to GCM."""
    import numpy as np
    import xarray as xr

    from src.celestials.planets.mars import mcd

    local_times = ([float(local_time)] if local_time is not None
                   else [float(hour) for hour in range(0, 24, 2)])
    samples = []
    sources = []
    altitude_m = float(fields.approximate_wind_height_m)
    for hour in local_times:
        key = (round(float(ls_deg), 4), hour, int(dust), round(altitude_m, 3))
        if key not in _mcd_ascii_cache:
            def token(value: float) -> str:
                return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")

            stem = (
                f"mcd_v6p1_ls{token(float(ls_deg))}_lt{token(hour)}_dust{int(dust)}"
                f"_z{token(altitude_m)}m_highres"
            )
            cache_path = _BENCHMARK_DATA_DIR / f"{stem}.txt"
            provenance_path = _BENCHMARK_DATA_DIR / f"{stem}.json"
            if cache_path.exists():
                text = cache_path.read_text()
                if provenance_path.exists():
                    source = json.loads(provenance_path.read_text()).get(
                        "source", f"cached:{cache_path}"
                    )
                else:
                    source = f"cached:{cache_path}"
                _mcd_ascii_cache[key] = (text, source)
            else:
                text, source = mcd.fetch_ascii(
                    ls_deg, hour, dust=dust, high_res=True, altitude_m=altitude_m,
                )
                _BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text)
                provenance_path.write_text(json.dumps({
                    "source": source, "mcd_version": "6.1", "ls_deg": float(ls_deg),
                    "local_time_hours": hour, "dust_scenario": int(dust),
                    "altitude_m_above_surface": altitude_m,
                    "high_resolution_topography": True,
                }, indent=2) + "\n")
                _mcd_ascii_cache[key] = (text, source)
        text, source = _mcd_ascii_cache[key]
        samples.append(mcd.parse_ascii(text))
        sources.append(source)
    native = xr.concat(
        samples, dim=xr.IndexVariable("local_time", local_times)
    ).mean("local_time")
    target = xr.Dataset(coords={"lat": fields.lat_deg, "lon": fields.lon_deg})
    matched = mcd.interpolate_periodic(native, target)

    model_arrays = {
        "temperature": np.asarray(fields.temperature_k),
        "surface_pressure": np.asarray(fields.surface_pressure_pa),
        "wind_speed": np.asarray(fields.wind_speed_ms),
        "co2_ice": np.asarray(fields.co2_ice_pa),
    }
    labels = {
        "temperature": ("Surface temperature", "K"),
        "surface_pressure": ("Surface pressure", "Pa"),
        "wind_speed": ("Horizontal wind speed", "m/s"),
        "co2_ice": ("CO2 surface frost", "Pa-equiv"),
    }

    def grid(values, label, units):
        values = np.asarray(values, dtype=float)
        return {
            "label": label, "units": units,
            "min": float(np.nanmin(values)), "max": float(np.nanmax(values)),
            "data": np.round(values, 3).tolist(),
        }

    mcd_maps = {}
    difference_maps = {}
    metrics = {}
    for name, model in model_arrays.items():
        reference = np.asarray(matched[name])
        label, units = labels[name]
        mcd_maps[name] = grid(reference, f"MCD {label}", units)
        difference_maps[name] = grid(
            model - reference, f"GCM − MCD {label}", units
        )
        metrics[name] = mcd.weighted_metrics(
            model, reference, np.asarray(fields.lat_deg)
        )
    return {
        "mcd": mcd_maps,
        "difference": difference_maps,
        "metrics": metrics,
        "metadata": {
            "mcd_version": "6.1", "ls_deg": float(ls_deg),
            "local_times_hours": local_times, "dust_scenario": int(dust),
            "wind_altitude_m": altitude_m, "source_urls": sources,
            "cache_directory": str(_BENCHMARK_DATA_DIR),
            "status": "diagnostic_transient_vs_climatology",
        },
    }


def _mcd_comparison_for_maps(fields: dict, maps: dict, ls_deg: float,
                             local_time: float | None, dust: int) -> dict:
    """Adapt browser field grids back to the MCD comparison's numeric contract."""
    from types import SimpleNamespace
    import numpy as np
    from src.celestials.planets.mars import MARS_BODY_3D

    n_layers = max(1, int(fields["metadata"].get("n_layers", 1) or 1))
    wind_height = -(
        MARS_BODY_3D.gas_constant_j_kg_k * MARS_BODY_3D.reference_temperature_k
        / MARS_BODY_3D.gravity_m_s2
    ) * np.log(1.0 - 0.5 / n_layers)
    model_snapshot = SimpleNamespace(
        lat_deg=np.asarray(fields["lat"]), lon_deg=np.asarray(fields["lon"]),
        temperature_k=np.asarray(maps["surface_temperature"]["data"]),
        surface_pressure_pa=np.asarray(maps["surface_pressure"]["data"]),
        wind_speed_ms=np.asarray(maps["surface_wind_speed"]["data"]),
        co2_ice_pa=np.asarray(maps["co2_ice"]["data"]),
        approximate_wind_height_m=wind_height,
    )
    return _matched_mcd_comparison(model_snapshot, ls_deg, local_time, dust)


def _attach_mcd_benchmark(fields: dict, ls_deg: float,
                          local_time: float | None, dust: int) -> list[float]:
    """Attach MCD to headline fields or fixed-local-time diurnal snapshots."""
    diurnal = fields.get("diurnal")
    if not diurnal:
        fields["comparison"] = _mcd_comparison_for_maps(
            fields, fields["maps"], ls_deg, local_time, dust
        )
        return ([] if local_time is None else [float(local_time)])

    available = [float(hour) for hour in diurnal["local_times_hours"]]
    targets = available
    if local_time is not None:
        targets = [min(available, key=lambda hour: abs((hour - local_time + 12) % 24 - 12))]
    for hour in targets:
        snapshot = diurnal["snapshots"][str(int(hour) if hour.is_integer() else hour)]
        snapshot["comparison"] = _mcd_comparison_for_maps(
            fields, snapshot["maps"], ls_deg, hour, dust
        )
    return targets


def _uploaded_netcdf_comparison(model_fields: dict, payload: bytes, filename: str,
                                local_time_h: float | None = None) -> dict:
    """Align an uploaded Ames-compatible NetCDF to a completed GCM run."""
    import tempfile
    import numpy as np
    import xarray as xr
    from src.celestials.planets.mars import mcd

    aliases = {
        "surface_temperature": ("surface_temperature", "temperature", "tsurf", "ts"),
        "surface_pressure": ("surface_pressure", "ps", "psurf"),
        "surface_wind_speed": ("wind_speed", "surface_wind_speed", "wind"),
        "co2_ice": ("co2_ice", "co2ice", "ice"),
        "elevation": ("elevation", "topography", "orog"),
    }
    suffix = Path(filename).suffix or ".nc"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(payload)
        tmp.flush()
        ds = xr.open_dataset(tmp.name).load()
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError("NetCDF needs lat/latitude and lon/longitude coordinates")
    ds = ds.rename({lat_name: "lat", lon_name: "lon"})
    ds = ds.assign_coords(lon=np.mod(ds.lon, 360.0)).sortby("lon").sortby("lat")
    target_lat = np.asarray(model_fields["lat"])
    target_lon = np.asarray(model_fields["lon"])

    def grid(values, label, units):
        a = np.asarray(values, dtype=float)
        return {"label": label, "units": units, "min": float(np.nanmin(a)),
                "max": float(np.nanmax(a)), "data": np.round(a, 3).tolist()}

    reference_maps, difference_maps, metrics = {}, {}, {}
    for model_name, candidates in aliases.items():
        if model_name not in model_fields["maps"]:
            continue
        variable = next((name for name in candidates if name in ds.data_vars), None)
        if variable is None:
            continue
        da = ds[variable].squeeze(drop=True)
        time_dim = next((name for name in ("local_time", "time_of_day", "lt")
                         if name in da.dims), None)
        if time_dim is not None and local_time_h is not None:
            axis = np.asarray(da[time_dim], dtype=float)
            index = int(np.argmin(np.abs((axis - local_time_h + 12.0) % 24.0 - 12.0)))
            da = da.isel({time_dim: index})
        if {"lat", "lon"} - set(da.dims):
            continue
        while da.ndim > 2:
            da = da.isel({da.dims[0]: -1})
        reference = np.asarray(da.interp(lat=target_lat, lon=target_lon).transpose("lat", "lon"))
        model_grid = model_fields["maps"][model_name]
        model = np.asarray(model_grid["data"])
        units = str(da.attrs.get("units", model_grid["units"]))
        reference_maps[model_name] = grid(reference, f"Ames {model_grid['label']}", units)
        difference_maps[model_name] = grid(model - reference, f"DMGCM − Ames {model_grid['label']}", model_grid["units"])
        metrics[model_name] = mcd.weighted_metrics(model, reference, target_lat)
    if not metrics:
        raise ValueError("No comparable 2-D temperature, pressure, wind, ice, or elevation fields found")
    return {"reference": reference_maps, "difference": difference_maps,
            "metrics": metrics, "metadata": {"filename": filename, "kind": "AmesGCM NetCDF",
            **({"local_time_h": float(local_time_h)} if local_time_h is not None else {})}}
def _gcm_snapshot(scale: str, albedo: float, greenhouse: float, ls_deg: float,
                  pressure_pa: float | None, diurnal: bool = False,
                  compare_mcd: bool = False, mcd_local_time: float | None = None,
                  mcd_dust: int = 1, surface_temp_k: float | None = None,
                  duration_sols: float | None = None,
                  progress_callback=None, diagnostic_callback=None,
                  stop_requested=None) -> dict:
    """Run one 3-D gcm3d map at the given atmosphere state; return field grids.

    ``pressure_pa`` (if given) sets the reference surface pressure, so a snapshot
    taken partway through a terraforming run reflects that epoch's evolved (thicker)
    atmosphere and greenhouse factor. With ``diurnal`` the run uses the moving
    day/night terminator; the timestep is then capped to the terminator CFL
    (dt <= rotation/(2*n_lon)) and the step count raised to hold physical duration,
    so a diurnal request never trips the guard.
    """
    import dataclasses
    import math

    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars.maps import resolve_scale, run_maps
    from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
    from src.framework.physics.gcm import mean_anomaly_for_ls

    forcing = radiative_forcing(
        albedo=albedo, greenhouse_factor=greenhouse, diurnal=diurnal,
        co2_radiation_enabled=True,
    )
    forcing = dataclasses.replace(
        forcing, init_orbital_angle_rad=mean_anomaly_for_ls(math.radians(ls_deg), forcing),
    )
    cfg = resolve_scale(scale)
    if duration_sols is not None:
        cfg["n_steps"] = max(1, round(
            duration_sols * forcing.rotation_period_s / cfg["dt_seconds"]
        ))
    if diurnal:
        n_lon = len(coordinate_system(cfg["truncation"], n_layers=1).horizontal.longitudes)
        max_dt = forcing.rotation_period_s / (2.0 * n_lon)
        if cfg["dt_seconds"] > max_dt:
            factor = math.ceil(cfg["dt_seconds"] / max_dt)
            cfg["dt_seconds"] = cfg["dt_seconds"] / factor
            cfg["n_steps"] = cfg["n_steps"] * factor
    fields = run_maps(
        forcing=forcing, co2_forcing=co2_forcing(),
        p0_pa=pressure_pa, t_ref_k=surface_temp_k,
        progress_callback=progress_callback, diagnostic_callback=diagnostic_callback,
        stop_requested=stop_requested,
        progress_chunk_steps=(max(1, cfg["n_steps"] // 48) if diurnal else 32),
        **cfg,
    )
    extracted = _extract_maps_fields(fields)
    if compare_mcd:
        extracted["comparison"] = _matched_mcd_comparison(
            fields, ls_deg, mcd_local_time, mcd_dust
        )
    return extracted


def _run_gcm_maps(run, req: RunRequest, cfg) -> None:
    """Run a single gcm3d 3-D map (no timeseries) and attach the field grids.

    ``run["data"]`` stays empty, so the browser shows the ``FieldMap`` view.
    """
    p = cfg.planet
    started = time.monotonic()
    raw_diurnal = []

    def progress(completed, total):
        fraction = completed / total
        elapsed = time.monotonic() - started
        run["progress"] = fraction
        run["completed_steps"] = completed
        run["total_steps"] = total
        run["elapsed_seconds"] = elapsed
        run["eta_seconds"] = elapsed * (1.0 - fraction) / fraction if fraction else None

    def sample_global_maps(completed, total, state, coords, specs):
        from src.celestials.planets.mars import MARS_BODY_3D
        raw_diurnal.append(_state_map_snapshot(state, coords, specs, MARS_BODY_3D))

    run["fields"] = _gcm_snapshot(
        req.scale, p.albedo, p.greenhouse_factor, p.initial_ls_deg or 0.0,
        pressure_pa=p.surface_pressure, diurnal=req.diurnal,
        compare_mcd=req.compare_mcd and not req.diurnal,
        mcd_local_time=req.mcd_local_time,
        mcd_dust=req.mcd_dust,
        surface_temp_k=p.surface_temperature,
        duration_sols=req.sols,
        progress_callback=progress,
        diagnostic_callback=sample_global_maps if req.diurnal else None,
        stop_requested=lambda: bool(run.get("cancel_requested")),
    )
    if req.diurnal:
        from src.celestials.planets.mars import MARS_BODY_3D
        run["fields"]["diurnal"] = _assemble_fixed_local_time_maps(
            raw_diurnal, run["fields"], MARS_BODY_3D.rotation_period_s
        )
        if req.compare_mcd:
            _attach_mcd_benchmark(
                run["fields"], p.initial_ls_deg or 0.0, None, req.mcd_dust
            )
    run["progress"] = 1.0


def _snapshot_years(n_years: int, n_snapshots: int) -> set[int]:
    """The intervention years at which to capture a 3-D snapshot.

    Evenly spaced across 1..n_years, always including the final year (so the last
    snapshot is the terraformed end-state used as the headline heatmap).
    """
    if n_snapshots <= 0 or n_years <= 0:
        return set()
    if n_snapshots >= n_years:
        return set(range(1, n_years + 1))
    import numpy as np
    years = {int(y) for y in np.linspace(1, n_years, n_snapshots).round().astype(int)}
    years.add(n_years)
    return years


def _run_simulation(run_id: str, req: RunRequest) -> None:
    """Execute the simulation in a background thread.

    Every step writes one data point to _runs[run_id]["data"].  The SSE
    generator reads that list with a cursor so the browser sees live updates.

    List.append is GIL-safe in CPython; no explicit lock is needed.
    """
    run = _runs[run_id]
    try:
        from cli import config_loader
        from cli.models import Accuracy, ExpType, RunFlags
        from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
        from src.engine import Accuracy as SrcAccuracy, TimeController
        from src.interventions import InterventionController

        flags = RunFlags(
            exp_type=ExpType(req.exp_type),
            accuracy=Accuracy(req.accuracy),
            dt=req.dt,
            lat=req.lat,
            lon=req.lon,
            elevation=req.elevation,
            ls=req.ls,
            surface_temp=req.surface_temp,
            pressure=req.surface_pressure,
            albedo=req.albedo,
            greenhouse_factor=req.greenhouse_factor,
            ice_mass=req.ice_mass,
            sols=req.sols if req.exp_type == "sol" else None,
            n_years=req.years if req.exp_type == "intervention" else None,
            inject=req.inject or None,
            no_save=True,
        )
        cfg = config_loader.load(planet="mars", preset=req.preset)
        cfg = config_loader.merge_overrides(cfg, flags)
        p = cfg.planet

        # The 'gcm' accuracy runs the 3-D gcm3d maps backend (dinosaur dycore +
        # radiation + CO2). A one-off gcm map (no timeseries) short-circuits here;
        # a gcm *intervention* runs the 100-yr trajectory and captures 3-D snapshots
        # along that existing timeline (handled below, after Mars is built).
        if req.accuracy == "gcm" and req.exp_type != "intervention":
            _run_gcm_maps(run, req, cfg)
            run["status"] = "done"
            run["progress"] = 1.0
            run["completed_at"] = datetime.now(timezone.utc).isoformat()
            return

        mars = Mars(
            surface_temperature=p.surface_temperature,
            surface_pressure=p.surface_pressure,
            albedo=p.albedo,
            greenhouse_factor=p.greenhouse_factor,
            ice_mass=p.ice_mass,
            latitude=p.latitude,
            longitude=p.longitude,
            elevation_m=p.elevation_m,
            initial_ls_deg=p.initial_ls_deg,
        )
        # gcm-flavoured intervention drives the (torch) trajectory with the FAST
        # reduced-order kernel and layers 3-D snapshots on top.
        accuracy = SrcAccuracy.ACCURATE if req.accuracy == "accurate" else SrcAccuracy.FAST

        if req.exp_type == "intervention":
            _run_intervention(run, req, mars, cfg, accuracy,
                              capture_gcm=(req.accuracy == "gcm"))
        else:
            _run_timeseries(run, req, mars, cfg, accuracy)

        run["status"] = "done"
        run["progress"] = 1.0
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save_run_csv(run_id, run)

    except InterruptedError:
        run["status"] = "stopped"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        run["eta_seconds"] = None
    except Exception as exc:
        import traceback
        run["status"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _run_intervention(run, req, mars, cfg, accuracy, capture_gcm: bool = False) -> None:
    """One data point per Mars year via InterventionController callback.

    When ``capture_gcm`` is set, a 3-D gcm3d snapshot is also captured at a handful
    of years along this same (existing) timeline — using each year's evolved
    pressure + greenhouse factor — so the browser can show the spatial evolution
    of the terraforming run, with the final year as the headline heatmap.
    """
    from src.interventions import InterventionController

    ic = InterventionController(
        mars,
        injection_schedule_kg_yr=req.inject or {},
        dt=cfg.engine.dt,
        accuracy=accuracy,
    )
    n_years = req.years
    ls_deg = cfg.planet.initial_ls_deg or 0.0
    snap_years = _snapshot_years(n_years, req.snapshots) if capture_gcm else set()
    if capture_gcm:
        run["snapshot_years"] = sorted(snap_years)
        run["field_snapshots"] = {}
        run["snapshot_errors"] = {}  # year -> traceback, surfaced to the browser

    def iv_cb(snap) -> None:
        run["data"].append({
            "year": snap.year,
            "temperature_k": _v(snap.surface_temperature),
            "temp_min_k":    _v(snap.temp_min),
            "temp_max_k":    _v(snap.temp_max),
            "pressure_pa":   _v(snap.surface_pressure),
            "ice_mass_kg":   _v(snap.ice_mass),
            "delta_F":       _v(snap.delta_F),
            "greenhouse_factor": _v(snap.greenhouse_factor),
        })
        # 3-D snapshot at this year's evolved atmosphere (thicker air + stronger
        # greenhouse as the terraforming proceeds).
        if snap.year in snap_years:
            try:
                fld = _gcm_snapshot(
                    req.scale,
                    albedo=cfg.planet.albedo,
                    greenhouse=_v(snap.greenhouse_factor),
                    ls_deg=ls_deg,
                    pressure_pa=_v(snap.surface_pressure),
                    diurnal=req.diurnal,
                    compare_mcd=req.compare_mcd,
                    mcd_local_time=req.mcd_local_time,
                    mcd_dust=req.mcd_dust,
                    surface_temp_k=_v(snap.surface_temperature),
                )
                run["field_snapshots"][str(snap.year)] = fld
                run["fields"] = fld  # latest successful snapshot is the headline
            except Exception:  # noqa: BLE001 — record the failure, do not hide it
                import traceback
                run["snapshot_errors"][str(snap.year)] = traceback.format_exc()
        # Progress: trajectory year plus the extra weight of snapshot rendering.
        run["progress"] = snap.year / n_years

    ic.run(n_years=n_years, callback=iv_cb)

    if capture_gcm:
        errors = run["snapshot_errors"]
        # The final-year snapshot is the required headline map: if it failed, the
        # GCM job failed. Earlier snapshot failures are partial-success.
        if str(n_years) in errors:
            raise RuntimeError(
                f"required final-year (Ls end) 3-D snapshot failed:\n"
                f"{errors[str(n_years)]}"
            )
        if errors:
            run["partial"] = True
            run["warning"] = (
                f"{len(errors)} of {len(snap_years)} snapshots failed "
                f"(years {sorted(int(y) for y in errors)}); see snapshot_errors."
            )


def _run_timeseries(run, req, mars, cfg, accuracy) -> None:
    """Per-step callback throttled to MAX_CHART_POINTS for sol/year runs.

    The TimeController calls callback(planet, elapsed) after every integration
    step.  We emit every Nth step so the browser receives at most
    MAX_CHART_POINTS data points regardless of run length.
    """
    from src.celestials import MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
    from src.engine import TimeController

    sol_s = float(MARS_ROTATION_PERIOD.item())

    if req.exp_type == "year":
        duration = float(MARS_ORBITAL_PERIOD.item()) * req.years
    else:
        duration = cfg.experiment.sols * sol_s

    dt_s = float(cfg.engine.dt)
    total_steps = max(1, int(duration / dt_s))
    emit_every = max(1, total_steps // MAX_CHART_POINTS)

    tc = TimeController(mars, dt=dt_s, accuracy=accuracy)
    step_n = [0]

    orbital_s = float(MARS_ORBITAL_PERIOD.item())

    def step_cb(planet, elapsed) -> None:
        step_n[0] += 1
        if step_n[0] % emit_every != 0:
            return
        t_s = _v(elapsed)
        pt: dict = {
            "time_h":        t_s / 3600.0,
            "sol":           t_s / sol_s,
            "temperature_k": _v(planet.thermal.surface_temperature),
            "pressure_pa":   _v(planet.atmosphere.surface_pressure),
            "ice_mass_kg":   _v(planet.water.ice_mass),
            "solar_flux":    _v(planet.radiation.solar_flux),
        }
        if req.exp_type == "year":
            pt["mars_year"] = t_s / orbital_s
        run["data"].append(pt)
        run["progress"] = step_n[0] / total_steps

    tc.run(duration=duration, callback=step_cb)


# ── API routes ─────────────────────────────────────────────────────────────────

@app.post("/api/runs")
async def create_run(req: RunRequest) -> dict:
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "id":           run_id,
        "status":       "running",
        "progress":     0.0,
        "config":       req.model_dump(),
        "data":         [],
        "error":        None,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "label":        req.label or _default_label(req),
        "cancel_requested": False,
        "completed_steps": 0,
        "total_steps": 0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
    }
    _executor.submit(_run_simulation, run_id, req)
    return {"run_id": run_id}


@app.post("/api/runs/import/netcdf")
async def import_run_netcdf(request: Request) -> dict:
    """Register an existing DMGCM NetCDF as a completed benchmarkable run."""
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "Empty NetCDF upload")
    filename = request.headers.get("x-filename", "imported-dmgcm.nc")
    try:
        fields = _imported_netcdf_fields(payload, filename)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    run_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    label = Path(filename).stem
    _runs[run_id] = {
        "id": run_id, "status": "done", "progress": 1.0,
        "config": {"preset": "imported", "exp_type": "sol", "accuracy": "gcm",
                   "sols": fields["metadata"]["duration_sols"], "scale": "imported",
                   "ls": None, "diurnal": "diurnal" in fields},
        "data": [], "fields": fields, "error": None, "created_at": now,
        "completed_at": now, "label": label, "cancel_requested": False,
        "completed_steps": 0, "total_steps": 0, "elapsed_seconds": 0.0,
        "eta_seconds": None, "imported_from": filename,
    }
    return {"run_id": run_id, "label": label}


@app.get("/api/runs")
async def list_runs() -> list:
    return [
        {k: v for k, v in run.items() if k not in {"data", "fields", "field_snapshots"}}
        for run in reversed(list(_runs.values()))
    ]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    # Field grids (3-D maps) are served separately so run polling stays light.
    heavy = {"fields", "field_snapshots"}
    return {k: v for k, v in _runs[run_id].items() if k not in heavy}


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    """Request cooperative cancellation at the next bounded GCM chunk."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if run["status"] != "running":
        return {"status": run["status"]}
    run["cancel_requested"] = True
    return {"status": "stopping"}


@app.get("/api/runs/{run_id}/fields")
async def get_run_fields(run_id: str, year: int | None = None) -> dict:
    """The lat/lon field grids for a gcm run.

    With no ``year`` this returns the headline snapshot (the final/most-recent one).
    Pass ``year`` to fetch a specific snapshot from a terraforming timeline.
    """
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if year is not None:
        snap = (run.get("field_snapshots") or {}).get(str(year))
        if not snap:
            raise HTTPException(404, f"No snapshot for year {year}")
        return snap
    fields = run.get("fields")
    if not fields:
        raise HTTPException(404, "No field data (not a gcm run, or not finished)")
    return fields


@app.get("/api/runs/{run_id}/snapshots")
async def get_run_snapshots(run_id: str) -> dict:
    """The years for which 3-D snapshots exist (for the timeline slider)."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    years = sorted(int(y) for y in (run.get("field_snapshots") or {}))
    return {"years": years}


@app.get("/api/runs/{run_id}/netcdf")
async def download_run_netcdf(run_id: str) -> Response:
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    fields = _runs[run_id].get("fields")
    if not fields:
        raise HTTPException(409, "The DMGCM run has no completed fields")
    payload = _fields_netcdf_bytes(fields)
    return Response(payload, media_type="application/x-netcdf",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}.nc"'})


@app.post("/api/runs/{run_id}/benchmarks/ames")
async def upload_ames_benchmark(run_id: str, request: Request) -> dict:
    """Attach a local AmesGCM NetCDF comparison without requiring multipart."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    fields = run.get("fields")
    if not fields:
        raise HTTPException(409, "The DMGCM run must finish before comparison")
    payload = await request.body()
    if not payload:
        raise HTTPException(400, "Empty NetCDF upload")
    filename = request.headers.get("x-filename", "ames-output.nc")
    try:
        comparison = _uploaded_netcdf_comparison(fields, payload, filename)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    fields.setdefault("benchmarks", {})["ames"] = comparison
    for hour, snapshot in (fields.get("diurnal") or {}).get("snapshots", {}).items():
        snapshot_fields = {"lat": fields["lat"], "lon": fields["lon"],
                           "maps": snapshot["maps"]}
        snapshot.setdefault("benchmarks", {})["ames"] = _uploaded_netcdf_comparison(
            snapshot_fields, payload, filename, float(hour)
        )
    return comparison


@app.post("/api/runs/{run_id}/benchmarks/mcd")
async def configure_mcd_benchmark(run_id: str, request: MCDBenchmarkRequest) -> dict:
    """Fetch and attach an automatically matched or explicitly configured MCD run."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    fields = run.get("fields")
    if not fields:
        raise HTTPException(409, "The DMGCM run must finish before comparison")
    config = run.get("config", {})
    ls_deg = float(config.get("ls") or 0.0) if request.auto_match else float(request.ls or 0.0)
    if request.ls is not None:
        ls_deg = float(request.ls)
    if not 0.0 <= ls_deg <= 360.0:
        raise HTTPException(422, "Ls must be between 0 and 360 degrees")
    if not 1 <= request.dust <= 8:
        raise HTTPException(422, "MCD dust scenario must be between 1 and 8")
    try:
        local_times = _attach_mcd_benchmark(
            fields, ls_deg, request.local_time, request.dust
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(502, f"MCD fetch failed: {exc}") from exc
    run["mcd_benchmark"] = {
        "ls_deg": ls_deg, "dust_scenario": request.dust,
        "local_times_hours": local_times, "auto_matched": request.auto_match,
    }
    return run["mcd_benchmark"]


@app.get("/api/runs/{run_id}/events")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE stream — emits new data points as the simulation progresses.

    The generator polls _runs[run_id]["data"] with a cursor, batching up to
    100 points per tick.  Poll interval is 250 ms.  When the simulation thread
    sets status to "done" or "error" a final event is emitted and the stream
    closes.
    """
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")

    async def generator():
        cursor = 0
        last_progress = None
        while True:
            run  = _runs[run_id]
            data = run["data"]

            while cursor < len(data):
                batch = data[cursor:cursor + 100]
                for pt in batch:
                    yield f"data: {json.dumps({'type': 'point', 'data': pt})}\n\n"
                cursor += len(batch)

            progress = run.get("progress", 0.0)
            if progress != last_progress:
                yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'eta_seconds': run.get('eta_seconds'), 'completed_steps': run.get('completed_steps', 0), 'total_steps': run.get('total_steps', 0)})}\n\n"
                last_progress = progress

            if run["status"] in ("done", "error", "stopped"):
                yield f"data: {json.dumps({'type': 'done', 'status': run['status'], 'error': run['error']})}\n\n"
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/compounds")
async def get_compounds() -> list[str]:
    from src.interventions.compounds import list_compounds
    return list_compounds()


@app.get("/api/presets")
async def get_presets() -> list[str]:
    from cli.presets import MARS_PRESET_NAMES
    return list(MARS_PRESET_NAMES)


@app.get("/api/presets/{name}")
async def get_preset_config(name: str) -> dict:
    from cli.presets import MARS_PRESET_NAMES
    if name not in MARS_PRESET_NAMES:
        raise HTTPException(404, f"Unknown preset '{name}'")
    from cli import config_loader
    cfg = config_loader.load(planet="mars", preset=name)
    return {"planet": cfg.planet.model_dump(), "engine": cfg.engine.model_dump()}


# ── Static UI — registered last so API routes win ──────────────────────────────

if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
