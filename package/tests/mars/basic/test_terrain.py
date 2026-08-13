"""Tests for src/celestials/planets/mars/terrain.py.

Covers:
  - load_mola_orography rejects a file that is not a global 2:1 grid.
  - build_mars_terrain (flat) gives all-land terrain with zero height.
  - load_mola_orography regrids the real MOLA file to the model grid (slow,
    needs the downloaded data; skipped when it is absent).
  - build_mars_terrain (MOLA) gives a non-zero surface geopotential.

Run scripts/download_mola_topography.py to get the MOLA data for the slow tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.celestials.planets.mars.gcm import _apply_mars_constants
from src.celestials.planets.mars.terrain import (
    DEFAULT_MOLA_IMG,
    build_mars_terrain,
    load_mola_orography,
)

_HAS_MOLA = DEFAULT_MOLA_IMG.exists()
_needs_mola = pytest.mark.skipif(not _HAS_MOLA, reason="MOLA data not downloaded")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def coords():
    """The JCM horizontal grid used by the placeholder model."""
    from jcm.physics.held_suarez.utils import get_held_suarez_coords
    _apply_mars_constants()   # terrain geopotential uses Mars gravity
    return get_held_suarez_coords()


# ── load_mola_orography: input validation (fast, no data) ─────────────────────

class TestLoaderValidation:

    def test_rejects_non_global_grid(self, tmp_path):
        """A file that is not a 2:1 global grid raises ValueError."""
        bad = tmp_path / "bad.img"
        np.arange(100, dtype=">i2").tofile(bad)   # 100 is not 2*n^2
        with pytest.raises(ValueError, match="global 2:1 grid"):
            load_mola_orography(bad, coords=None)


# ── build_mars_terrain: flat placeholder (slow: builds spectral terrain) ──────

@pytest.mark.slow
class TestFlatTerrain:

    def test_is_all_land(self, coords):
        """Mars terrain is all land: fmask is 1 everywhere."""
        terrain = build_mars_terrain(coords, mola_img_path=None)
        assert float(np.asarray(terrain.fmask).min()) == pytest.approx(1.0)

    def test_flat_height_is_zero(self, coords):
        """With no MOLA file the orography is flat."""
        terrain = build_mars_terrain(coords, mola_img_path=None)
        assert float(np.abs(np.asarray(terrain.orog)).max()) == pytest.approx(0.0)


# ── load_mola_orography: the real regrid (slow, needs data) ───────────────────

@pytest.mark.slow
@_needs_mola
class TestMolaRegrid:

    def test_shape_is_ix_il(self, coords):
        """The regridded height has the JCM terrain shape (nlon, nlat)."""
        orog = load_mola_orography(DEFAULT_MOLA_IMG, coords)
        n_lon = np.asarray(coords.horizontal.longitudes).shape[0]
        n_lat = np.asarray(coords.horizontal.latitudes).shape[0]
        assert tuple(orog.shape) == (n_lon, n_lat)

    def test_is_finite(self, coords):
        """Every regridded height is finite."""
        orog = np.asarray(load_mola_orography(DEFAULT_MOLA_IMG, coords))
        assert bool(np.all(np.isfinite(orog)))

    def test_elevation_range_is_mars_like(self, coords):
        """The coarse grid keeps a deep basin and a high volcano."""
        orog = np.asarray(load_mola_orography(DEFAULT_MOLA_IMG, coords))
        assert orog.min() < -3000.0    # Hellas basin
        assert orog.max() > 5000.0     # Tharsis / Olympus

    def test_terrain_geopotential_is_nonzero(self, coords):
        """Real topography gives a non-zero surface geopotential (g * orog)."""
        terrain = build_mars_terrain(coords, mola_img_path=DEFAULT_MOLA_IMG)
        assert float(np.abs(np.asarray(terrain.phis0)).max()) > 0.0
