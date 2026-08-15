"""Tests for src.gcm3d.maps — dry-dynamics Mars maps over MOLA terrain
(requires the optional 'gcm3d' extra and the staged MOLA raster).

Covers:
  - hydrostatic_surface_pressure_pa: Mars barometry (sea-level = p0, sign).
  - run_maps: field shapes, finiteness, physical ranges, and the key
    invariant — surface pressure anti-correlates with terrain elevation.
  - save_netcdf / plot_maps: output files are written.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.gcm3d import maps  # noqa: E402
from src.gcm3d import topography as topo  # noqa: E402

jax.config.update("jax_enable_x64", True)

needs_mola = pytest.mark.skipif(
    not topo._DEFAULT_MOLA.exists(), reason="MOLA MEGDR raster not staged"
)


class TestHydrostaticPressure:

    def test_sea_level_equals_reference(self):
        p = maps.hydrostatic_surface_pressure_pa(0.0, MARS_BODY_3D)
        assert float(p) == pytest.approx(MARS_BODY_3D.reference_surface_pressure_pa)

    def test_higher_terrain_lower_pressure(self):
        low = maps.hydrostatic_surface_pressure_pa(-4000.0, MARS_BODY_3D)
        high = maps.hydrostatic_surface_pressure_pa(20000.0, MARS_BODY_3D)
        assert float(high) < MARS_BODY_3D.reference_surface_pressure_pa < float(low)
        # ~10 km scale height ⇒ Olympus (20 km) near ~85 Pa.
        assert 60.0 < float(high) < 120.0


@needs_mola
class TestRunMaps:

    @pytest.fixture(scope="class")
    def fields(self):
        # T21 / few layers / short run keeps this fast but exercises the whole path.
        return maps.run_maps(
            truncation="T21", n_layers=8, dt_seconds=600.0, n_steps=40
        )

    def test_field_shapes(self, fields):
        n_lon = fields.lon_deg.size
        n_lat = fields.lat_deg.size
        for arr in (
            fields.elevation_m,
            fields.surface_pressure_pa,
            fields.temperature_k,
            fields.u_ms,
            fields.v_ms,
        ):
            assert arr.shape == (n_lat, n_lon)

    def test_fields_finite_and_physical(self, fields):
        assert np.all(np.isfinite(fields.surface_pressure_pa))
        assert np.all(fields.surface_pressure_pa > 0.0)
        assert np.all(fields.temperature_k > 100.0) and np.all(fields.temperature_k < 350.0)
        assert np.all(np.isfinite(fields.wind_speed_ms))

    def test_surface_pressure_tracks_topography(self, fields):
        # The defining validation signature: low p over high terrain
        # (Tharsis/Olympus), high p in Hellas. Expect strong anti-correlation.
        r = np.corrcoef(
            fields.elevation_m.ravel(), fields.surface_pressure_pa.ravel()
        )[0, 1]
        assert r < -0.9

    def test_run_maps_rejects_nonpositive_steps(self):
        with pytest.raises(ValueError):
            maps.run_maps(truncation="T21", n_layers=8, n_steps=0)


@needs_mola
class TestOutputs:

    def test_save_netcdf_and_plots(self, tmp_path):
        f = maps.run_maps(truncation="T21", n_layers=8, dt_seconds=600.0, n_steps=20)
        nc = maps.save_netcdf(f, tmp_path / "mars_maps.nc")
        assert nc.exists()
        import xarray as xr

        ds = xr.open_dataset(nc)
        assert set(["surface_pressure", "temperature", "u", "v", "elevation"]).issubset(ds.data_vars)
        ds.close()

        pngs = maps.plot_maps(f, tmp_path, prefix="mars")
        assert len(pngs) == 4
        for p in pngs:
            assert p.exists() and p.stat().st_size > 0
