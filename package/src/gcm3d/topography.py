"""MOLA Mars topography → spectral orography for the gcm3d dycore.

Loads the MOLA MEGDR global topography (PDS ``.img`` raster staged at
``data/mola/meg004/megt90n000cb.img``: 720×1440, 16-bit MSB signed integers,
metres, simple-cylindrical) and turns it into the modal orography the primitive
equations take as their lower boundary — so a Mars run happens over the *real*
terrain (Tharsis, Olympus Mons, Hellas, Valles Marineris) instead of a flat
sphere. This is what makes the output maps comparable in form to the Ames MGCM
and LMD PCM, which both run on Mars topography.

Pure-NumPy raster read + bilinear regrid; the only JAX/dinosaur touch is the
final spherical-harmonic transform, taken from the guarded ``_dinosaur`` entry
point. Requires the optional ``gcm3d`` extra.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.gcm3d._dinosaur import jnp, primitive_equations, scales

_u = scales.units

# ── MOLA MEGDR provenance (authoritative source + integrity) ──────────────────
# Product: MGS MOLA MEGDR global topography, 4 pixels/degree (megt90n000cb).
# Node:    NASA PDS Geosciences Node, dataset MGS-M-MOLA-5-MEGDR-L3-V1.
# License: public domain (U.S. Government / NASA PDS); cite Smith et al. (2001),
#          JGR 106, "Mars Orbiter Laser Altimeter: Experiment summary...".
# The URL may move between PDS mirrors; integrity is guaranteed by the SHA-256
# below (verified regardless of source), so a wrong/old URL fails loudly at
# staging rather than silently loading corrupt data.
MOLA_SOURCE_URL = (
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/meg/megt90n000cb.img"
)
MOLA_LABEL_URL = MOLA_SOURCE_URL[:-4] + ".lbl"
MOLA_SHA256 = "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
MOLA_SIZE_BYTES = 2_073_600  # 720 × 1440 × 2 (int16)

# Staging location: env override MOLA_PATH, else repo-relative default.
_REPO_DEFAULT_MOLA = (
    Path(__file__).resolve().parents[3]
    / "data" / "mola" / "meg004" / "megt90n000cb.img"
)


def _default_mola_path() -> Path:
    """Configured MOLA path: ``$MOLA_PATH`` if set, else the repo default."""
    env = os.environ.get("MOLA_PATH")
    return Path(env) if env else _REPO_DEFAULT_MOLA


# Back-compat module constant (tests reference topography._DEFAULT_MOLA).
_DEFAULT_MOLA = _default_mola_path()

# MEGDR ``megt90n000cb`` raster geometry (from the PDS .lbl).
_MOLA_LINES = 720      # latitude rows,  0.25° each
_MOLA_SAMPLES = 1440   # longitude cols, 0.25° each
_MOLA_DEG_PER_PX = 0.25


def verify_mola_checksum(path: str | Path | None = None) -> str:
    """Return the SHA-256 of the staged raster, raising if it doesn't match.

    Guards against truncated/corrupt downloads and wrong-product files — the
    numbers only mean anything if the bytes are exactly the MEGDR product.
    """
    import hashlib

    path = Path(path) if path is not None else _default_mola_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MOLA_SHA256:
        raise ValueError(
            f"MOLA raster at {path} has SHA-256 {digest}, expected {MOLA_SHA256}. "
            f"The file is corrupt or the wrong product; re-stage with "
            f"scripts/stage_mola.py."
        )
    return digest


def load_mola_meg(path: str | Path | None = None):
    """Load the MOLA MEGDR raster as ``(elevation_m, lats_deg, lons_deg)``.

    Returns
    -------
    elevation_m : np.ndarray, shape (720, 1440)
        Elevation in metres (planetary radius minus areoid), row 0 = +89.875°N,
        column 0 = 0.125°E. Documented data range is −8068 m (Hellas) to
        +21134 m (Olympus Mons).
    lats_deg : np.ndarray, shape (720,)   Descending pixel-centre latitudes.
    lons_deg : np.ndarray, shape (1440,)  Ascending pixel-centre longitudes [0, 360).
    """
    path = Path(path) if path is not None else _default_mola_path()
    if not path.exists():
        raise FileNotFoundError(
            f"MOLA MEGDR raster not found at {path}.\n"
            f"The 3-D Mars maps require the real MEGDR topography — this is a "
            f"setup step, not an optional skip.\n"
            f"  • Stage it:   python scripts/stage_mola.py\n"
            f"  • Or set MOLA_PATH=/absolute/path/to/megt90n000cb.img\n"
            f"  • Source:      {MOLA_SOURCE_URL}\n"
            f"  • SHA-256:     {MOLA_SHA256}"
        )
    # 16-bit big-endian (MSB) signed integers, metres.
    raw = np.fromfile(path, dtype=">i2")
    if raw.size != _MOLA_LINES * _MOLA_SAMPLES:
        raise ValueError(
            f"MOLA raster has {raw.size} samples, expected "
            f"{_MOLA_LINES * _MOLA_SAMPLES} (720×1440)."
        )
    elevation_m = raw.reshape(_MOLA_LINES, _MOLA_SAMPLES).astype(np.float64)

    half = _MOLA_DEG_PER_PX / 2.0
    lats_deg = 90.0 - (np.arange(_MOLA_LINES) + 0.5) * _MOLA_DEG_PER_PX  # +89.875..−89.875
    lons_deg = (np.arange(_MOLA_SAMPLES) + 0.5) * _MOLA_DEG_PER_PX       # 0.125..359.875
    del half
    return elevation_m, lats_deg, lons_deg


def _bilinear_periodic(elev, src_lats_deg, src_lons_deg, tgt_lat_deg, tgt_lon_deg):
    """Bilinearly sample ``elev`` at target lat/lon (degrees).

    Longitude wraps periodically at 360°; latitude is clamped to the raster
    span. ``src_lats_deg`` is descending, ``src_lons_deg`` ascending — both
    uniformly spaced (MOLA MEGDR). Vectorised over the target arrays.
    """
    nlat, nlon = elev.shape
    dlat = _MOLA_DEG_PER_PX
    dlon = _MOLA_DEG_PER_PX

    # Latitude index (descending grid): fractional row from top (+90).
    fi = (src_lats_deg[0] - tgt_lat_deg) / dlat
    fi = np.clip(fi, 0.0, nlat - 1.0000001)
    i0 = np.floor(fi).astype(int)
    i1 = np.minimum(i0 + 1, nlat - 1)
    wi = fi - i0

    # Longitude index (ascending, periodic).
    fj = (np.mod(tgt_lon_deg, 360.0) - src_lons_deg[0]) / dlon
    fj = np.mod(fj, nlon)
    j0 = np.floor(fj).astype(int) % nlon
    j1 = (j0 + 1) % nlon
    wj = fj - np.floor(fj)

    v00 = elev[i0, j0]
    v01 = elev[i0, j1]
    v10 = elev[i1, j0]
    v11 = elev[i1, j1]
    top = v00 * (1 - wj) + v01 * wj
    bot = v10 * (1 - wj) + v11 * wj
    return top * (1 - wi) + bot * wi


def regrid_to_nodal(coords, elevation_m=None, mola_path=None) -> np.ndarray:
    """Regrid MOLA elevation onto the dynamics nodal grid.

    Returns a NumPy array of elevation in metres shaped ``coords.horizontal
    .nodal_shape`` (``(n_lon, n_lat)``), ready to nondimensionalize. Bilinear;
    the dycore's own spectral truncation handles smoothing to the model
    resolution afterwards.
    """
    if elevation_m is None:
        elevation_m, src_lats, src_lons = load_mola_meg(mola_path)
    else:
        elevation_m = np.asarray(elevation_m, dtype=np.float64)
        src_lats = 90.0 - (np.arange(elevation_m.shape[0]) + 0.5) * _MOLA_DEG_PER_PX
        src_lons = (np.arange(elevation_m.shape[1]) + 0.5) * _MOLA_DEG_PER_PX

    lon_mesh, sin_lat_mesh = coords.horizontal.nodal_mesh  # each (n_lon, n_lat)
    tgt_lon_deg = np.degrees(np.asarray(lon_mesh))
    tgt_lat_deg = np.degrees(np.arcsin(np.asarray(sin_lat_mesh)))
    return _bilinear_periodic(elevation_m, src_lats, src_lons, tgt_lat_deg, tgt_lon_deg)


def mola_modal_orography(coords, specs, elevation_nodal_m=None, mola_path=None,
                         wavenumbers_to_clip: int = 1):
    """Build the modal orography lower boundary from MOLA topography.

    Steps: regrid MOLA → nodal metres, nondimensionalize the height with
    ``specs``, then spherical-harmonic transform with the top ``wavenumbers_to_clip``
    total wavenumbers clipped (``truncated_modal_orography``) to suppress Gibbs
    ringing at the truncation. Returns a modal array of shape
    ``coords.horizontal.modal_shape`` — pass it straight to
    :func:`src.gcm3d.primitive_equations` as ``orography=…``.
    """
    if elevation_nodal_m is None:
        elevation_nodal_m = regrid_to_nodal(coords, mola_path=mola_path)
    height_nd = specs.nondimensionalize(np.asarray(elevation_nodal_m) * _u.meter)
    return primitive_equations.truncated_modal_orography(
        jnp.asarray(height_nd), coords, wavenumbers_to_clip=wavenumbers_to_clip
    )
