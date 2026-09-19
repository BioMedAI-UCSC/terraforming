#!/usr/bin/env python3
"""Stage independent MGS/TES albedo and thermal inertia into a 1-degree NetCDF."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

ALBEDO_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"
INERTIA_URL = (
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/"
    "mgst_9001/data/global_ti_night_2007.img"
)
ALBEDO_SHA256 = "c91dfcaa1834b96db383beddec537d85e8e82b782682744ee723907ed9e58178"
INERTIA_SHA256 = "faee32838da40045ee7ebd21bed156cc649620312c01165b556d77a2128258f4"


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "terraforming-data-stage/1"})
    with urllib.request.urlopen(request, timeout=300) as source, path.open("wb") as target:  # noqa: S310
        while chunk := source.read(1024 * 1024):
            target.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _block_mean(values: np.ndarray, pixels_per_degree: int) -> np.ndarray:
    rows, columns = values.shape
    blocks = values.reshape(
        rows // pixels_per_degree, pixels_per_degree,
        columns // pixels_per_degree, pixels_per_degree,
    )
    count = np.sum(np.isfinite(blocks), axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def main() -> None:
    raw = Path("data/tes/raw")
    albedo_path = raw / "Mars_MGS_TES_Albedo_mosaic_global_7410m.tif"
    inertia_path = raw / "global_ti_night_2007.img"
    _download(ALBEDO_URL, albedo_path)
    _download(INERTIA_URL, inertia_path)
    for path, expected in (
        (albedo_path, ALBEDO_SHA256),
        (inertia_path, INERTIA_SHA256),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")

    Image.MAX_IMAGE_PIXELS = None
    albedo = np.asarray(Image.open(albedo_path), dtype=np.float64)
    if albedo.shape != (1440, 2880):
        raise ValueError(f"unexpected TES albedo shape {albedo.shape}")
    albedo[(albedo < 0.0) | (albedo > 1.0)] = np.nan
    albedo = _block_mean(albedo, 8)

    inertia = np.memmap(inertia_path, dtype=">i2", mode="r", shape=(3600, 7200))
    inertia = np.asarray(inertia, dtype=np.float64)
    inertia[(inertia < 5.0) | (inertia > 5000.0)] = np.nan
    inertia = _block_mean(inertia, 20)

    # Both products are north-to-south and -180..180 east longitude.
    albedo = np.roll(albedo[::-1], 180, axis=1)
    inertia = np.roll(inertia[::-1], 180, axis=1)
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    ds = xr.Dataset(
        {
            "albedo": (("lat", "lon"), albedo, {"units": "1"}),
            "thermal_inertia": (
                ("lat", "lon"), inertia,
                {"units": "J m-2 K-1 s-1/2"},
            ),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "title": "Independent MGS TES surface boundary fields",
            "albedo_source": ALBEDO_URL,
            "thermal_inertia_source": INERTIA_URL,
            "albedo_sha256": ALBEDO_SHA256,
            "thermal_inertia_sha256": INERTIA_SHA256,
            "aggregation": "area-unweighted 1-degree block mean on native cylindrical grids",
        },
    )
    output = Path("data/tes/mgs_tes_surface_1deg.nc")
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output)
    print(output)
    print(f"albedo_sha256={ds.attrs['albedo_sha256']}")
    print(f"thermal_inertia_sha256={ds.attrs['thermal_inertia_sha256']}")
    print(f"albedo={float(ds.albedo.min()):.3f}..{float(ds.albedo.max()):.3f}")
    print(f"thermal_inertia={float(ds.thermal_inertia.min()):.1f}..{float(ds.thermal_inertia.max()):.1f}")


if __name__ == "__main__":
    main()
