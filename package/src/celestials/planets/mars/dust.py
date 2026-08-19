"""Prescribed Ames seasonal dust climatology for the differentiable Mars GCM."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def seasonal_dust_on_grid(grid, path: str | Path, ls_deg: float):
    """Interpolate Ames column opacity and dust-top height to a Dinosaur grid."""
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dust scenario file not found: {path}")
    with xr.open_dataset(path) as opened:
        ds = opened[["tau", "zmax"]].assign_coords(areo=("time", opened.areo.values)).swap_dims(
            {"time": "areo"}
        ).load()
    # Drop the duplicate 360-degree endpoint and add periodic guards.
    ds = ds.isel(areo=np.asarray(ds.areo) < 359.999)
    before = ds.isel(areo=-1).assign_coords(areo=float(ds.areo[-1]) - 360.0)
    after = ds.isel(areo=0).assign_coords(areo=float(ds.areo[0]) + 360.0)
    ds = xr.concat([before, ds, after], dim="areo")
    west = ds.isel(lon=-1).assign_coords(lon=float(ds.lon[-1]) - 360.0)
    east = ds.isel(lon=0).assign_coords(lon=float(ds.lon[0]) + 360.0)
    ds = xr.concat([west, ds, east], dim="lon")
    result = ds.interp(
        areo=float(ls_deg) % 360.0,
        lon=np.degrees(np.asarray(grid.longitudes)) % 360.0,
        lat=np.degrees(np.asarray(grid.latitudes)),
    )
    tau = np.asarray(result.tau).T
    zmax = np.asarray(result.zmax).T
    if not np.isfinite(tau).all() or not np.isfinite(zmax).all():
        raise ValueError(f"dust scenario {path} produced missing values")
    return tau, zmax
