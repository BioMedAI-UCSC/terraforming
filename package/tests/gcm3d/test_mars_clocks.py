"""The dynamical and solar clocks must remain independently specified."""
import dataclasses
import math

import numpy as np
import pytest

from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.framework.gcm.body import EARTH


def test_sidereal_coriolis_and_solar_duration():
    assert BODY.rotation_period_s == pytest.approx(88642.663)
    assert BODY.angular_velocity_s == pytest.approx(2 * math.pi / 88642.663)
    assert BODY.solar_day_seconds == pytest.approx(88775.244)
    assert EARTH.solar_day_seconds == 86400.
    legacy = dataclasses.replace(BODY, solar_day_s=None)
    assert legacy.solar_day_seconds == legacy.rotation_period_s


@pytest.mark.parametrize("invalid", [0., -1., float('inf'), float('nan')])
def test_invalid_solar_clock(invalid):
    with pytest.raises(ValueError, match="solar_day_s"):
        dataclasses.replace(BODY, solar_day_s=invalid)


def test_noon_repeats_after_solar_sol_not_sidereal_day():
    pytest.importorskip("dinosaur")
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.celestials.planets.mars.gcm import radiative_forcing
    from src.framework.physics.gcm import cos_zenith_nodal
    from src.framework.gcm.coordinates import coordinate_system
    coords = coordinate_system("T21", 4)
    # Freeze orbital phase so this checks only the hour-angle clock.
    f = dataclasses.replace(radiative_forcing(diurnal=True), orbital_period_s=1e30)
    assert f.rotation_period_s == BODY.solar_day_seconds
    initial = np.asarray(cos_zenith_nodal(0., coords.horizontal.latitudes, coords.horizontal.longitudes, f))
    solar = np.asarray(cos_zenith_nodal(BODY.solar_day_seconds, coords.horizontal.latitudes, coords.horizontal.longitudes, f))
    sidereal = np.asarray(cos_zenith_nodal(BODY.rotation_period_s, coords.horizontal.latitudes, coords.horizontal.longitudes, f))
    np.testing.assert_allclose(solar, initial, atol=1e-12)
    assert np.max(np.abs(sidereal-initial)) > 1e-4


def test_map_export_records_both_clocks_and_reports_solar_sols(tmp_path):
    pytest.importorskip("dinosaur")
    import xarray as xr
    from src.celestials.planets.mars.maps import MarsMapFields, save_netcdf
    zeros = np.zeros((2, 2))
    fields = MarsMapFields(
        lon_deg=np.array([0., 180.]), lat_deg=np.array([-45., 45.]),
        elevation_m=zeros, surface_pressure_pa=zeros+610., temperature_k=zeros+200.,
        u_ms=zeros, v_ms=zeros, truncation="T21", n_layers=4,
        n_steps=1, dt_seconds=88775.244, physics="clock fixture",
    )
    assert fields.duration_sols == pytest.approx(1.)
    with xr.open_dataset(save_netcdf(fields, tmp_path / "clocks.nc")) as ds:
        assert ds.attrs["solar_day_s"] == pytest.approx(88775.244)
        assert ds.attrs["sidereal_rotation_period_s"] == pytest.approx(88642.663)
