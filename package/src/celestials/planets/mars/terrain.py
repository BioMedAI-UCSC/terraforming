"""Mars terrain for the JCM backend.

This module turns a MOLA MEGDR topography grid into JCM ``TerrainData``.  The
MOLA height is the ``orog`` field.  Mars has no oceans, so the land-sea mask is
1 everywhere.

Get the MOLA data with ``scripts/download_mola_topography.py``.  The default file
is the 4 pixels-per-degree global grid, at ``data/mola/meg004/megt90n000cb.img``.

The MOLA ``.img`` format (MEGDR topography):
  - Big-endian 16-bit signed integers, in metres.
  - The height is relative to the Mars areoid (the equipotential reference).
  - Row 0 is 90°N.  The rows go north to south.
  - Column 0 is 0°E.  The columns go west to east (0–360).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jcm.terrain import TerrainData

# The default MOLA file, relative to the repository root.
#   terrain.py → mars → planets → celestials → src → package → repo
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MOLA_IMG = _REPO_ROOT / "data" / "mola" / "meg004" / "megt90n000cb.img"


def load_mola_orography(img_path: Path, coords) -> jnp.ndarray:
    """Read a MOLA ``.img`` grid and regrid it onto the JCM horizontal grid.

    Parameters
    ----------
    img_path : Path
        The MOLA MEGDR topography ``.img`` file.
    coords : CoordinateSystem
        The JCM coordinates.  The target grid is ``coords.horizontal``.

    Returns
    -------
    jnp.ndarray
        The surface height in metres, shape ``(ix, il)`` = (nlon, nlat).
    """
    # 1. Read the raw heights.  Infer the global 2:1 grid from the file size.
    raw = np.fromfile(img_path, dtype=">i2").astype(np.float64)
    n_lat = int(round((raw.size / 2) ** 0.5))
    n_lon = 2 * n_lat
    if n_lat * n_lon != raw.size:
        raise ValueError(f"MOLA file size {raw.size} is not a global 2:1 grid")
    grid = raw.reshape(n_lat, n_lon)   # [lat: 90N→90S, lon: 0→360E]

    # 2. Build the source pixel-center coordinates in degrees.
    d_lat = 180.0 / n_lat
    d_lon = 360.0 / n_lon
    src_lat = 90.0 - d_lat * (np.arange(n_lat) + 0.5)   # descending
    src_lon = d_lon * (np.arange(n_lon) + 0.5)          # 0→360

    # 3. Flip latitude to ascending; the interpolator needs ascending axes.
    src_lat_asc = src_lat[::-1]
    grid_asc = grid[::-1, :]

    # 4. Pad the longitude axis so it is periodic (0 and 360 wrap correctly).
    lon_ext = np.concatenate([[src_lon[-1] - 360.0], src_lon, [src_lon[0] + 360.0]])
    grid_ext = np.concatenate([grid_asc[:, -1:], grid_asc, grid_asc[:, :1]], axis=1)

    interp = RegularGridInterpolator(
        (src_lat_asc, lon_ext), grid_ext,
        method="linear", bounds_error=False,
        fill_value=None,  # type: ignore[arg-type]  # None = extrapolate; scipy stub types this as float
    )

    # 5. Evaluate on the model grid (radians → degrees).
    tgt_lat = np.degrees(np.asarray(coords.horizontal.latitudes))    # (nlat_m,)
    tgt_lon = np.degrees(np.asarray(coords.horizontal.longitudes))   # (nlon_m,)
    lon_mesh, lat_mesh = np.meshgrid(tgt_lon, tgt_lat, indexing="xy")  # (nlat_m, nlon_m)
    points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
    height = interp(points).reshape(lat_mesh.shape)   # (nlat_m, nlon_m)

    # 6. Transpose to (ix, il) = (nlon, nlat), the JCM terrain shape.
    return jnp.asarray(height.T)


def build_mars_terrain(coords, mola_img_path: Path | None = None) -> TerrainData:
    """Build Mars ``TerrainData`` for the JCM model.

    Mars is all land, so ``fmask`` is 1 everywhere and ``lfluxland`` is True.

    Parameters
    ----------
    coords : CoordinateSystem
        The JCM coordinates.
    mola_img_path : Path or None
        The MOLA topography file.  If None, the terrain is flat (height 0).
        Use flat terrain only as a placeholder.

    Returns
    -------
    TerrainData
        The terrain for ``Model(terrain=...)``.
    """
    n_lon = int(np.asarray(coords.horizontal.longitudes).shape[0])
    n_lat = int(np.asarray(coords.horizontal.latitudes).shape[0])

    if mola_img_path is not None:
        orography = load_mola_orography(mola_img_path, coords)
    else:
        orography = jnp.zeros((n_lon, n_lat))   # flat placeholder

    fmask = jnp.ones((n_lon, n_lat))   # all land
    return TerrainData.from_coords(
        coords,
        orography=orography,
        fmask=fmask,
        lfluxland=True,
    )
