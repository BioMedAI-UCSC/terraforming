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


class TestScalePresets:

    def test_known_presets_resolve(self):
        for name in ("fast", "balanced", "high", "ultra"):
            cfg = maps.resolve_scale(name)
            assert set(cfg) == {"truncation", "n_layers", "dt_seconds", "n_steps"}
            assert cfg["n_layers"] > 0 and cfg["n_steps"] > 0

    def test_default_and_ordering(self):
        # Higher scales are at least as fine as lower ones.
        fast = maps.resolve_scale("fast")
        ultra = maps.resolve_scale("ultra")
        assert ultra["n_layers"] >= fast["n_layers"]
        assert ultra["n_steps"] >= fast["n_steps"]
        assert maps.resolve_scale(None) == maps.resolve_scale(maps.DEFAULT_SCALE)

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="unknown scale"):
            maps.resolve_scale("gigantic")


def test_explicit_surface_fields_match_grid_and_physical_bounds(tmp_path):
    from src.gcm3d.coordinates import coordinate_system
    from src.gcm3d.surface import surface_fields_on_grid
    import xarray as xr

    grid = coordinate_system("T21", 8).horizontal
    path = tmp_path / "surface.nc"
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(0.0, 330.0, 12)
    shape = (lat.size, lon.size)
    xr.Dataset(
        {
            "albedo": (("lat", "lon"), np.full(shape, 0.25)),
            "thermal_inertia": (("lat", "lon"), np.full(shape, 250.0)),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    albedo, inertia = surface_fields_on_grid(grid, path)
    assert albedo.shape == grid.nodal_shape
    assert inertia.shape == grid.nodal_shape
    assert np.all((albedo >= 0.0) & (albedo <= 1.0))
    assert np.all(inertia > 0.0)


def test_surface_fields_are_threaded_into_radiation_and_regolith(tmp_path):
    import xarray as xr

    from src.gcm3d.coordinates import coordinate_system
    from src.gcm3d.physics import mars_radiative_forcing

    grid = coordinate_system("T21", 8).horizontal
    path = tmp_path / "surface.nc"
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(0.0, 330.0, 12)
    lon_pattern = np.linspace(0.1, 0.4, lon.size)[None, :]
    albedo = np.broadcast_to(lon_pattern, (lat.size, lon.size))
    inertia = np.broadcast_to(100.0 + 900.0 * lon_pattern, albedo.shape)
    xr.Dataset(
        {
            "albedo": (("lat", "lon"), albedo),
            "thermal_inertia": (("lat", "lon"), inertia),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)

    forcing = maps.forcing_with_surface_properties(
        mars_radiative_forcing(diurnal=False), grid, path
    )
    assert np.asarray(forcing.albedo).shape == grid.nodal_shape
    assert np.ptp(np.asarray(forcing.albedo)) > 0.2
    assert np.ptp(np.asarray(forcing.surface_thermal_inertia_tiu)) > 200.0
    assert forcing.regolith_enabled
    assert forcing.stability_exchange_enabled
    assert forcing.pbl_diffusion_enabled
    assert forcing.convective_adjustment_enabled


@needs_mola
class TestOutputs:

    def test_save_netcdf_and_plots(self, tmp_path):
        f = maps.run_maps(truncation="T21", n_layers=8, dt_seconds=600.0, n_steps=20)
        nc = maps.save_netcdf(f, tmp_path / "mars_maps.nc")
        assert nc.exists()
        import xarray as xr

        ds = xr.open_dataset(nc)
        assert set(["surface_pressure", "temperature", "u", "v", "elevation"]).issubset(ds.data_vars)
        assert 0.0 < ds.attrs["wind_level_sigma"] < 1.0
        assert ds.attrs["approximate_wind_height_m"] > 0.0
        ds.close()

        pngs = maps.plot_maps(f, tmp_path, prefix="mars")
        assert len(pngs) == 4
        for p in pngs:
            assert p.exists() and p.stat().st_size > 0
