#!/usr/bin/env python3
"""Extract a compact, season-indexed surface reference from Ames MGCM output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

MARS_GRAVITY_M_S2 = 3.72076


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with xr.open_dataset(args.source, decode_times=False) as source:
        ls = np.mod(source.areo[:, 0].values, 360.0)
        order = np.argsort(ls)

        def seasonal(variable):
            return variable.isel(time=order).assign_coords(time=ls[order]).rename(time="ls")

        staged = xr.Dataset(
            {
                "surface_pressure": seasonal(source.ps),
                "temperature": seasonal(source.ts),
                "lowest_layer_air_temperature": seasonal(source.temp_bot),
                "u": seasonal(source.ucomp_bot),
                "v": seasonal(source.vcomp_bot),
                "co2_ice": seasonal(source.co2ice_sfc) * MARS_GRAVITY_M_S2,
                "dust_visible_optical_depth": seasonal(source.taudust_VIS),
                "dust_longwave_optical_depth": seasonal(source.taudust_IR),
                "average_duration_days": seasonal(source.average_DT),
            },
        ).load()
    staged["wind_speed"] = np.hypot(staged.u, staged.v)
    units = {
        "surface_pressure": "Pa", "temperature": "K",
        "lowest_layer_air_temperature": "K", "u": "m/s", "v": "m/s",
        "wind_speed": "m/s", "co2_ice": "Pa",
        "dust_visible_optical_depth": "1", "dust_longwave_optical_depth": "1",
        "average_duration_days": "day",
    }
    for name, unit in units.items():
        staged[name].attrs["units"] = unit
    staged.attrs.update(
        title="NASA Ames MGCM FV3 Beta Output Release 1 surface reference",
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_code_release="2022.02",
        source_simulation="c48 reference, sixth simulated year, 2-degree output",
        temporal_sampling="five-sol averages",
        co2_ice_conversion=f"kg/m2 multiplied by Mars gravity {MARS_GRAVITY_M_S2} m/s2",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged.to_netcdf(args.output, engine="h5netcdf")
    report = {
        "output": str(args.output), "source": str(args.source),
        "source_sha256": staged.attrs["source_sha256"],
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "samples": int(staged.sizes["ls"]),
        "ls_range_deg": [float(staged.ls.min()), float(staged.ls.max())],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
