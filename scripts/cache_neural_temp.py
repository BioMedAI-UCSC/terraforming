#!/usr/bin/env python3
"""Frozen physical forecasts, observed initialization and atomic per-start caches."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

import numpy as np

from neural_temp_data import digest, read_run, sha256, write_json, write_npz
from neural_temp_model import build_features


def physical_forcing():
    from src.celestials.planets.mars.gcm import radiative_forcing
    return dataclasses.replace(radiative_forcing(diurnal=True, co2_radiation_enabled=True),
                               regolith_enabled=True, stability_exchange_enabled=True,
                               pbl_diffusion_enabled=True, convective_adjustment_enabled=True)


def spinup_soil(history, forcing, interval):
    """Backward Euler finite-volume conduction; fixed observed upper boundary.

    The oldest observed surface T initializes every depth. Bottom flux is zero.
    Twelve intervals use their left boundary (past-only, piecewise constant).
    Geometry and conductivity match regolith_conduction_tendencies exactly.
    """
    history = np.asarray(history, dtype=float)
    if len(history) != 13 or not np.isfinite(history).all():
        raise ValueError("soil needs exactly one sol of finite past surface history")
    cv = forcing.regolith_volumetric_heat_capacity_j_m3_k
    inertia = float(forcing.surface_thermal_inertia_tiu)
    dz = np.maximum(np.asarray(forcing.regolith_layer_skin_depth_fractions) * inertia / cv * np.sqrt(forcing.rotation_period_s / np.pi), 1e-4)
    k = inertia**2 / cv
    n = len(dz)
    matrix = np.eye(n)
    top = k / (.5 * dz[0]) / (cv * dz[0])
    matrix[0, 0] += interval * top
    for j in range(n - 1):
        conductance = k / (.5 * (dz[j] + dz[j + 1]))
        for a, b in ((j, j + 1), (j + 1, j)):
            rate = interval * conductance / (cv * dz[a])
            matrix[a, a] += rate
            matrix[a, b] -= rate
    ground = np.broadcast_to(history[0], (n,) + history.shape[1:]).copy().reshape(n, -1)
    for surface in history[:-1]:
        rhs = ground.copy()
        rhs[0] += interval * top * surface.ravel()
        ground = np.linalg.solve(matrix, rhs)
    return ground.reshape((n,) + history.shape[1:])


def initialize(window, record, contract):
    # Imported only inside a masked worker, after device environment is set.
    from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic, primitive_equations
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm.dynamics import reference_temperature
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.framework.physics.gcm import ColumnPhysicsState, mean_anomaly_for_ls
    jax.config.update("jax_enable_x64", True)
    coords, specs = coordinate_system("T21", 12), physics_specs(body)
    grid, u = coords.horizontal, scales.units
    start = window.isel(time=12)
    def field(name, ds=start):
        dims = ("lev", "lon", "lat") if "lev" in ds[name].dims else ("lon", "lat")
        return np.asarray(ds[name].transpose(*dims))
    # MACDA epoch is midnight at lon=0; solver t=0 is local noon.
    ts = (record["start_sol"] + .5) * body.rotation_period_s
    forcing = physical_forcing()
    forcing = dataclasses.replace(forcing,
        init_orbital_angle_rad=mean_anomaly_for_ls(np.deg2rad(record["ls"]), forcing) - 2 * np.pi * ts / forcing.orbital_period_s,
        dust_visible_optical_depth=jnp.asarray(field("coldust")),
        dust_longwave_optical_depth=jnp.asarray(field("coldust") / 3))
    nd = lambda a, unit: jnp.asarray(specs.nondimensionalize(a * unit))
    vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(
        grid, nd(field("uwind"), u.meter/u.second), nd(field("vwind"), u.meter/u.second))
    state = primitive_equations.State(vorticity=vor, divergence=div,
        temperature_variation=grid.to_modal(nd(field("temp"), u.kelvin) - reference_temperature(coords, body)[:, None, None]),
        log_surface_pressure=grid.to_modal(jnp.log(nd(field("psurf")[None], u.pascal))),
        sim_time=nd(ts, u.second))
    history = np.asarray(window.tsurf.isel(time=slice(0, 13)).transpose("time", "lon", "lat"))
    soil = spinup_soil(history, forcing, contract["lead_seconds"])
    state = ColumnPhysicsState(state, nd(field("tsurf")[None], u.kelvin),
                               nd(field("co2ice")[None] * body.gravity_m_s2, u.pascal), nd(soil, u.kelvin))
    return coords, specs, forcing, state


def forecast(window, record, contract):
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.celestials.planets.mars.gcm import co2_forcing
    from src.celestials.planets.mars.topography import regrid_to_nodal
    from src.celestials.planets.mars.maps import run_maps, state_to_comparison_dataset
    from src.framework.physics.gcm import _true_anomaly, cos_zenith_nodal, solar_flux
    coords, specs, forcing, state = initialize(window, record, contract)
    grid = coords.horizontal
    ts = (record["start_sol"] + .5) * body.rotation_period_s
    start = window.isel(time=12)
    def field(name, ds=start):
        dims = ("lev", "lon", "lat") if "lev" in ds[name].dims else ("lon", "lat")
        return np.asarray(ds[name].transpose(*dims))
    _, final = run_maps(truncation="T21", n_layers=12, dt_seconds=contract["dt_seconds"],
        n_steps=contract["steps"], mola_path=contract["terrain_path"], forcing=forcing,
        co2_forcing=co2_forcing(), initial_state=state, return_final_state=True)
    out = state_to_comparison_dataset(final, coords, specs, body, forcing).isel(time=0)
    def f(name):
        return np.asarray(out[name].transpose(*(("sigma", "lon", "lat") if "sigma" in out[name].dims else ("lon", "lat"))))
    future = ts + contract["lead_seconds"]
    ls = float(_true_anomaly(future, forcing)) + forcing.ls_perihelion_rad
    hour = np.asarray(grid.longitudes)[:, None] + 2*np.pi*future/forcing.rotation_period_s
    cz = np.asarray(cos_zenith_nodal(future, grid.latitudes, grid.longitudes, forcing))
    temperature = f("air_temperature")
    fields = dict(start_t=field("temp")[-1], start_surface_t=field("tsurf"), start_next_t=field("temp")[-2],
        start_u=field("uwind")[-1], start_v=field("vwind")[-1], start_pressure=field("psurf"), start_dust=field("coldust"),
        forecast_t=temperature[-1], forecast_delta_t=temperature[-1]-field("temp")[-1],
        forecast_surface_t=f("surface_temperature"), forecast_surface_air_delta=f("surface_temperature")-temperature[-1],
        forecast_vertical_delta=temperature[-2]-temperature[-1], forecast_u=f("eastward_wind")[-1],
        forecast_v=f("northward_wind")[-1], forecast_pressure=f("surface_pressure"),
        insolation=cz*float(solar_flux(future, forcing)), sin_lst=np.sin(hour + np.pi), cos_lst=np.cos(hour + np.pi),
        sin_ls=np.sin(ls), cos_ls=np.cos(ls), sin_lat=np.sin(np.asarray(grid.latitudes))[None],
        sin_lon=np.sin(np.asarray(grid.longitudes))[:, None], cos_lon=np.cos(np.asarray(grid.longitudes))[:, None],
        terrain=regrid_to_nodal(coords, mola_path=contract["terrain_path"]))
    x = build_features(fields)
    weights = np.broadcast_to(np.polynomial.legendre.leggauss(x.shape[1])[1][None], x.shape[:2]).copy()
    weights /= weights.sum()
    if not np.any(cz > 0) or not np.any(cz == 0):
        raise ValueError("snapshot lacks geographic day/night coverage")
    return {"features": x.reshape(-1, x.shape[-1]),
            "target": field("temp", window.isel(time=13))[-1].ravel(),
            "area": weights.ravel(), "pressure": field("psurf").ravel(),
            "day": (cz > 0).ravel(), "lat": np.broadcast_to(np.degrees(grid.latitudes)[None], cz.shape).ravel(),
            "lon": np.broadcast_to(np.degrees(grid.longitudes)[:, None], cz.shape).ravel(),
            "lst": np.broadcast_to(np.mod(12 + 12*hour/np.pi, 24), cz.shape).ravel()}


def cache_entry(root, record, contract, arrays):
    folder = Path(root) / "cache" / record["id"]
    validate_arrays(arrays)
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite cache output")
    # Inference features and labels are deliberately separate files.
    y = arrays["target"]
    write_npz(folder / "features.npz", **{k: v for k, v in arrays.items() if k != "target"})
    write_npz(folder / "target.npz", target=y, physical_error=y-arrays["features"][:, 7])
    meta = {"contract_hash": digest(contract), "record_hash": digest(record),
            "files": {name: sha256(folder / name) for name in ("features.npz", "target.npz")}}
    write_json(folder / "complete.json", meta)  # Commit marker written last.


def validate_arrays(arrays):
    from neural_temp_model import SCHEMA
    required = {"features", "target", "area", "pressure", "day", "lat", "lon", "lst"}
    if not required <= arrays.keys():
        raise ValueError("incomplete cache fields")
    x = arrays["features"]
    if x.ndim != 2 or x.shape[1] != len(SCHEMA):
        raise ValueError("invalid feature schema/shape")
    if any(arrays[n].shape != (len(x),) for n in required-{"features"}):
        raise ValueError("unpaired cache shapes")
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite cache")
    if np.any(arrays["area"] <= 0) or not np.isclose(arrays["area"].sum(), 1) or np.any(arrays["pressure"] <= 0):
        raise ValueError("invalid area or pressure weights")


def read_entry(root, record, contract):
    folder = Path(root) / "cache" / record["id"]
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    if meta["contract_hash"] != digest(contract) or meta["record_hash"] != digest(record):
        raise ValueError("cache contract mismatch")
    if set(meta["files"]) != {"features.npz", "target.npz"}:
        raise ValueError("incomplete cache commit marker")
    result = {}
    for name, checksum in meta["files"].items():
        if sha256(folder / name) != checksum:
            raise ValueError("cache checksum mismatch")
        with np.load(folder / name, allow_pickle=False) as data:
            result.update({k: data[k] for k in data.files})
    validate_arrays(result)
    return result


def load_split(root, contract, manifest, split):
    records = [r for r in manifest["starts"] if r["split"] == split]
    rows = [read_entry(root, r, contract) for r in records]
    if len(records) != manifest["requested"][split] or any(r is None for r in rows):
        raise ValueError(f"incomplete {split} cache")
    return records, {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ids", nargs="+", required=True)
    args = p.parse_args()
    import xarray as xr
    from src.framework.gcm._dinosaur import jax
    devices = jax.devices()
    if os.environ.get("NEURAL_TEMP_GPU") and (len(devices) != 1 or devices[0].platform != "gpu"):
        raise RuntimeError(f"expected one masked GPU, got {devices}")
    print(f"GPU mask={os.environ.get('CUDA_VISIBLE_DEVICES')} actual={devices}", flush=True)
    contract, manifest = read_run(args.run_dir)
    if sha256(contract["terrain_path"]) != contract["terrain_sha256"]:
        raise ValueError("terrain changed")
    records = {r["id"]: r for r in manifest["starts"]}
    for index, name in enumerate(args.ids):
        record = records[name]
        if record["split"] == "test":
            from train_neural_temp import verify_frozen
            verify_frozen(args.run_dir, contract, require_gate=True)
        if read_entry(args.run_dir, record, contract) is not None:
            continue
        started = time.monotonic()
        path = args.run_dir / "native" / f"{name}.nc"
        if sha256(path) != record["native_sha256"]:
            raise ValueError("staged native window checksum changed")
        with xr.open_dataset(path, decode_times=False) as window:
            arrays = forecast(window.load(), record, contract)
        cache_entry(args.run_dir, record, contract, arrays)
        event = {"stage": "cache", "id": name, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "completed": index+1, "total": len(args.ids), "seconds": time.monotonic()-started}
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
