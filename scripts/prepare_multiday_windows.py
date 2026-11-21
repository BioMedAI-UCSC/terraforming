#!/usr/bin/env python3
"""Freeze five-sol forecasts with daily sampling and whole-year held-out splits."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import xarray as xr

from cache_neural_temp import initialize
from neural_temp_data import sha256, validate_units
from run_differentiable_experiments import d, p
from src.framework.gcm.restart import save_restart


def daily_sample_times(solar_day):
    samples = np.arange(1., 61.).reshape(5, 12) * solar_day / 12.
    centres = samples.mean(axis=1)
    dt = solar_day / 300.
    assert np.allclose(samples / dt, np.rint(samples / dt), rtol=0, atol=1e-9)
    assert np.allclose(samples.mean(axis=1), centres, rtol=0, atol=1e-8)
    assert samples.min() > 0. and np.isclose(samples.max(), 5. * solar_day, rtol=0, atol=1e-8)
    return centres, samples.ravel(), dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p.validate_inputs(args.inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    solar_day = d.MARS_BODY_3D.solar_day_seconds
    centres, samples, dt = daily_sample_times(solar_day)
    windows = []
    for year in (32, 33, 34, 35):
        split = "train" if year == 32 else "validation" if year == 33 else "test"
        for quarter, season in enumerate((45, 135, 225, 315)):
            path = args.daily / f"macda_my{year}_ls{season:03d}_10sol_daily_t21_l12.nc"
            with xr.open_dataset(path, decode_times=False) as opened:
                daily = opened.load()
            validate_units(daily)
            if daily.sizes != dict(time=10, lev=12, lat=32, lon=64):
                raise ValueError(f"Expected complete T21/L12 daily data: {path}")
            if not np.allclose(np.diff(daily.time), 1., rtol=0, atol=1e-8):
                raise ValueError("Daily targets must have exact one-sol cadence")
            if not np.all(np.rint(daily.MY_Ls) == year):
                raise ValueError("Daily windows must stay within their declared Mars year")
            # Only days 0/1 initialize the state and soil; days 2--6 are targets.
            history = daily.isel(time=slice(0, 2)).interp(time=np.linspace(float(daily.time[0]), float(daily.time[1]), 13))
            record = dict(start_sol=float(daily.time[1]), ls=float(daily.Ls[1]))
            contract = dict(lead_seconds=solar_day / 12., terrain_path=str((args.inputs / "mola.img").resolve()))
            _, specs, forcing, state = initialize(history, record, contract)
            # Start after the final sample contributing to the initial daily mean.
            # The mean fields estimate the state at that data-availability time.
            available_sol = record["start_sol"] + 5.5 / 12.
            time_shift = specs.nondimensionalize((5.5 / 12. * solar_day) * d.scales.units.second)
            state = state._replace(dynamics=dataclasses.replace(state.dynamics,
                sim_time=state.dynamics.sim_time + time_shift))
            stem = f"my{year}_q{quarter}_five_sol"
            restart, target, force = [args.output / (stem + suffix) for suffix in ("-restart.npz", "-target.npz", "-forcing.npz")]
            save_restart(state, restart)
            fields = np.stack([np.asarray(daily[name].isel(time=slice(2, 7)).transpose("time", "lev", "lon", "lat"))
                               for name in ("temp", "uwind", "vwind")], axis=1)
            if not np.isfinite(fields).all():
                raise ValueError("Nonfinite daily targets")
            np.savez_compressed(target, seconds=centres, sample_seconds=samples, fields=fields)
            np.savez_compressed(force, init_orbital_angle_rad=forcing.init_orbital_angle_rad,
                                dust_visible=forcing.dust_visible_optical_depth, dust_longwave=forcing.dust_longwave_optical_depth)
            start = available_sol * solar_day
            windows.append(dict(id=stem, split=split, block=f"my{year}_q{quarter}",
                source="ARCO-MACDA; doi:10.57967/hf/8771; native daily mean", reference_kind="reanalysis",
                source_file=path.name, source_file_sha256=sha256(path), source_license=str(daily.attrs.get("source_license", "unrecorded")),
                start_seconds=start, end_seconds=start + 5. * solar_day, samples_per_observation=12,
                restart=restart.name, target=target.name, forcing=force.name,
                restart_sha256=sha256(restart), target_sha256=sha256(target), forcing_sha256=sha256(force)))
    p.dump(args.output / "windows.json", dict(schema_version=1, windows=windows, dt_seconds=dt,
        terrain_sha256=sha256(args.inputs / "mola.img"), body_constants=dataclasses.asdict(d.MARS_BODY_3D),
        protocol="MY32 trains, MY33 selects, MY34/35 test; four seasons each; five daily-mean leads; 12 samples per mean.",
        limitation="Forecast starts after all initialization-day samples are available; daily means estimate that starting state. Soil uses one sol of linearly interpolated past daily means, without resolved diurnal history. Dust is held at its initial observed value. No initial-day sample overlaps target samples. Daily staging source revisions were not recorded; frozen input hashes identify the used files."))
    print(json.dumps(dict(windows=len(windows), train=4, validation=4, test=8, forecast_sols=5, dt_seconds=dt)))


if __name__ == "__main__":
    main()
