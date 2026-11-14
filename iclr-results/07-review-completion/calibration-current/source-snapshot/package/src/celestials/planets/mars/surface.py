"""Spatial Mars surface boundary fields used by deterministic column physics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

def surface_boundary_fields_on_grid(grid, path: str | Path):
    """Interpolate an explicitly supplied surface dataset to nodal arrays.

    The file must contain ``albedo`` and ``thermal_inertia`` on ``lat``/``lon``.
    No reference-climatology dataset is imported or selected implicitly.
    """
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"surface boundary file not found: {path}"
        )
    with xr.open_dataset(path) as opened:
        names = [name for name in ("albedo", "thermal_inertia", "emissivity", "roughness")
                 if name in opened]
        if not {"albedo", "thermal_inertia"}.issubset(names):
            raise ValueError(f"surface dataset {path} lacks albedo or thermal_inertia")
        ds = opened[names].load()
    ds = ds.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    ds = ds.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    ds = ds.assign_coords(lon=np.mod(ds.lon, 360.0)).sortby("lon")
    _, unique = np.unique(np.asarray(ds.lon), return_index=True)
    ds = ds.isel(lon=np.sort(unique))
    left = ds.isel(lon=-1).assign_coords(lon=float(ds.lon[-1]) - 360.0)
    right = ds.isel(lon=0).assign_coords(lon=float(ds.lon[0]) + 360.0)
    ds = xr.concat([left, ds, right], dim="lon")
    result = ds.interp(
        lon=np.degrees(np.asarray(grid.longitudes)) % 360.0,
        lat=np.degrees(np.asarray(grid.latitudes)),
    )
    albedo = np.asarray(result.albedo).T
    inertia = np.asarray(result.thermal_inertia).T
    if not np.isfinite(albedo).all() or not np.isfinite(inertia).all():
        raise ValueError(f"surface dataset {path} contains unfillable missing values")
    if not np.all((albedo >= 0.0) & (albedo <= 1.0)) or not np.all(inertia > 0.0):
        raise ValueError(f"surface dataset {path} contains nonphysical values")
    fields = {"albedo": albedo, "thermal_inertia": inertia}
    for name in ("emissivity", "roughness"):
        if name in result:
            value = np.asarray(result[name]).T
            if not np.isfinite(value).all() or np.any(value <= 0.0):
                raise ValueError(f"surface dataset {path} has invalid {name}")
            fields[name] = value
    return fields


def surface_fields_on_grid(grid, path: str | Path):
    """Backward-compatible albedo/thermal-inertia pair."""
    fields = surface_boundary_fields_on_grid(grid, path)
    return fields["albedo"], fields["thermal_inertia"]
