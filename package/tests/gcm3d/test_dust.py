"""Periodic dust interpolation without requiring an external Ames checkout."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dinosaur")

from src.framework.gcm.coordinates import coordinate_system  # noqa: E402
from src.celestials.planets.mars.dust import seasonal_dust_on_grid  # noqa: E402


AMES_DUST = Path(__file__).resolve().parents[3] / "AmesGCM/data/DustScenario_Background.nc"


@pytest.fixture(scope="module")
def dust_scenario(tmp_path_factory):
    import xarray as xr

    path = tmp_path_factory.mktemp("dust") / "scenario.nc"
    ls = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(0.0, 330.0, 12)
    tau = (0.3 + 0.1 * np.cos(np.radians(ls))[:, None, None]
           + 0.04 * np.sin(np.radians(lat))[None, :, None]
           + 0.02 * np.cos(np.radians(lon))[None, None, :])
    xr.Dataset(
        {
            "tau": (("time", "lat", "lon"), tau),
            "zmax": (("time", "lat", "lon"), np.full_like(tau, 30.0)),
            "areo": ("time", ls),
        },
        coords={"time": np.arange(ls.size), "lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


def test_ames_seasonal_dust_is_finite_periodic_and_spatial(dust_scenario):
    grid = coordinate_system(truncation="T21", n_layers=6).horizontal
    tau0, zmax0 = seasonal_dust_on_grid(grid, dust_scenario, 0.0)
    tau360, zmax360 = seasonal_dust_on_grid(grid, dust_scenario, 360.0)
    assert tau0.shape == grid.nodal_shape
    assert np.isfinite(tau0).all() and np.isfinite(zmax0).all()
    assert tau0.min() > 0.0 and np.ptp(tau0) > 0.0
    assert np.allclose(tau0, tau360)
    assert np.allclose(zmax0, zmax360)


def test_dust_changes_with_season(dust_scenario):
    grid = coordinate_system(truncation="T21", n_layers=6).horizontal
    tau0, _ = seasonal_dust_on_grid(grid, dust_scenario, 0.0)
    tau270, _ = seasonal_dust_on_grid(grid, dust_scenario, 270.0)
    assert np.max(np.abs(tau270 - tau0)) > 0.01


@pytest.mark.skipif(not AMES_DUST.exists(), reason="external Ames dust dataset not staged")
def test_staged_ames_dust_is_finite_and_periodic():
    grid = coordinate_system(truncation="T21", n_layers=6).horizontal
    tau, zmax = seasonal_dust_on_grid(grid, AMES_DUST, 0.0)
    tau360, zmax360 = seasonal_dust_on_grid(grid, AMES_DUST, 360.0)
    assert np.isfinite(tau).all() and np.isfinite(zmax).all()
    np.testing.assert_allclose(tau, tau360)
    np.testing.assert_allclose(zmax, zmax360)
