"""A located cold cell must survive the map reduction and coordinate order."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyse_polar_control import frost_deficit_map


def test_frost_map_preserves_landmark_and_rejects_bad_units_and_values():
    coords = dict(time=[15.], sigma=[.2, .8], lat=[-60., 0., 60.], lon=[0., 90., 180., 270.])
    pressure = np.full((1, 2, 3, 4), 600.)
    frost = 3182.48 / (23.3494 - np.log(6.))
    air = np.full_like(pressure, frost + 1.)
    air[0, 0, 2, 1] = frost - 7.
    ds = xr.Dataset({"air_temperature": (("time", "sigma", "lat", "lon"), air, {"units": "K"}),
                     "air_pressure": (("time", "sigma", "lat", "lon"), pressure, {"units": "Pa"})}, coords=coords)
    field = frost_deficit_map(ds.transpose("lon", "time", "lat", "sigma"))
    assert field.dims == ("lat", "lon")
    assert field.attrs["units"] == "K"
    assert field.sel(lat=60., lon=90.) == pytest.approx(7.)
    assert field.sum() == pytest.approx(7.)
    ds.air_pressure.attrs["units"] = "hPa"
    with pytest.raises(ValueError, match="pascal"):
        frost_deficit_map(ds)
    ds.air_pressure.attrs["units"] = "Pa"
    ds.air_temperature.values[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite positive"):
        frost_deficit_map(ds)
