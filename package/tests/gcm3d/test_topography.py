"""Tests for src.gcm3d.topography — MOLA MEGDR → spectral orography
(requires the optional 'gcm3d' extra and the staged MOLA raster).

Covers:
  - load_mola_meg: raster shape, axis conventions, documented elevation range.
  - _bilinear_periodic: exact at grid nodes, periodic in longitude.
  - regrid_to_nodal: model-grid shape, range bounded by the source.
  - mola_modal_orography: finite modal field of the expected shape.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.gcm3d import coordinate_system, physics_specs  # noqa: E402
from src.gcm3d import topography as topo  # noqa: E402

jax.config.update("jax_enable_x64", True)

_HAS_MOLA = topo._DEFAULT_MOLA.exists()
needs_mola = pytest.mark.skipif(not _HAS_MOLA, reason="MOLA MEGDR raster not staged")


class TestProvenance:
    """Integrity/reproducibility of the staged MOLA raster (fast, no dinosaur run)."""

    def test_env_path_override(self, monkeypatch, tmp_path):
        p = tmp_path / "custom.img"
        monkeypatch.setenv("MOLA_PATH", str(p))
        assert topo._default_mola_path() == p

    def test_missing_raster_gives_actionable_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MOLA_PATH", str(tmp_path / "nope.img"))
        with pytest.raises(FileNotFoundError, match="stage_mola"):
            topo.load_mola_meg()

    @needs_mola
    def test_staged_raster_matches_pinned_checksum(self):
        assert topo.verify_mola_checksum() == topo.MOLA_SHA256

    @needs_mola
    def test_corrupt_raster_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.img"
        bad.write_bytes(b"\x00" * topo.MOLA_SIZE_BYTES)
        with pytest.raises(ValueError, match="expected"):
            topo.verify_mola_checksum(bad)


@needs_mola
class TestLoadMola:

    def test_shape_axes_and_range(self):
        elev, lats, lons = topo.load_mola_meg()
        assert elev.shape == (720, 1440)
        # Descending latitudes from +89.875, ascending longitudes from 0.125.
        assert lats[0] == pytest.approx(89.875)
        assert lats[-1] == pytest.approx(-89.875)
        assert lons[0] == pytest.approx(0.125)
        assert lons[-1] == pytest.approx(359.875)
        # PDS-documented global extremes: Hellas −8068 m, Olympus Mons 21134 m.
        assert elev.min() == pytest.approx(-8068, abs=1)
        assert elev.max() == pytest.approx(21134, abs=1)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            topo.load_mola_meg("/nonexistent/mola.img")


class TestBilinear:

    def test_exact_at_nodes(self):
        # 4×8 synthetic raster; sampling a pixel centre returns that pixel.
        elev = np.arange(32, dtype=float).reshape(4, 8)
        lats = 90.0 - (np.arange(4) + 0.5) * topo._MOLA_DEG_PER_PX
        lons = (np.arange(8) + 0.5) * topo._MOLA_DEG_PER_PX
        # Note: helper assumes MOLA spacing; use matching lat/lon for node hits.
        for i in (0, 2, 3):
            for j in (0, 5, 7):
                got = topo._bilinear_periodic(
                    elev, lats, lons, np.array([lats[i]]), np.array([lons[j]])
                )
                assert got[0] == pytest.approx(elev[i, j])

    def test_longitude_is_periodic(self):
        elev = np.arange(32, dtype=float).reshape(4, 8)
        lats = 90.0 - (np.arange(4) + 0.5) * topo._MOLA_DEG_PER_PX
        lons = (np.arange(8) + 0.5) * topo._MOLA_DEG_PER_PX
        a = topo._bilinear_periodic(elev, lats, lons, np.array([0.0]), np.array([0.1]))
        b = topo._bilinear_periodic(elev, lats, lons, np.array([0.0]), np.array([360.1]))
        assert a[0] == pytest.approx(b[0])


@needs_mola
class TestRegridAndModal:

    @pytest.fixture(scope="class")
    def coords(self):
        return coordinate_system("T21", 10)

    def test_regrid_shape_and_bounds(self, coords):
        nod = topo.regrid_to_nodal(coords)
        assert nod.shape == coords.horizontal.nodal_shape
        # Bilinear stays within the source data envelope.
        assert nod.min() >= -8068 - 1
        assert nod.max() <= 21134 + 1
        assert np.all(np.isfinite(nod))

    def test_modal_orography_finite(self, coords):
        specs = physics_specs(MARS_BODY_3D)
        oro = topo.mola_modal_orography(coords, specs)
        assert oro.shape == coords.horizontal.modal_shape
        assert bool(np.all(np.isfinite(np.asarray(oro))))
        # Non-flat: real terrain carries power beyond the (0,0) mean mode.
        assert float(np.abs(np.asarray(oro)).sum()) > 0.0
