"""Seasonal maps must compare labelled grids and mean wind speeds."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from evaluate_arco_seasonal import align_reference, load_model_window, load_arco_window, weighted_metrics, wind_at_height


def test_reference_alignment_handles_longitude_wrap_and_reversed_latitude():
    reference = xr.Dataset({"temp": (("lev", "lon", "lat"),
        np.array([[[11., -9.], [12., -8.], [11., -9.], [10., -10.]]]))},
        coords=dict(lev=np.array([.5], dtype=np.float32), lon=[-180., -90., 0., 90.], lat=[10., -10.]))
    model = xr.Dataset({"air_temperature": (("sigma", "lat", "lon"),
        [[[-9., -10., -9., -8.], [11., 10., 11., 12.]]])},
        coords=dict(sigma=[.5], lat=[-10., 10.], lon=[0., 90., 180., 270.]))
    aligned = align_reference(reference, model)
    assert weighted_metrics(model.air_temperature, aligned.temp, model.lat)["rmse"] == 0.
    with pytest.raises(ValueError, match="coverage"):
        weighted_metrics(model.air_temperature, aligned.temp * np.nan, model.lat)


def test_window_selection_includes_each_time_and_speed_before_average(tmp_path):
    shape = (3, 1, 2, 2)
    wind = np.broadcast_to(np.array([2., -2., 100.])[:, None, None, None], shape)
    dims = ("time", "sigma", "lat", "lon")
    ds = xr.Dataset(dict(eastward_wind=(dims, wind), northward_wind=(dims, np.zeros(shape)),
                         solar_longitude=("time", [359., 1., 90.])),
                    coords=dict(time=[0, 1, 2], sigma=[.5], lat=[-10., 10.], lon=[0., 180.]))
    model_path = tmp_path / "model.nc"
    ds.to_netcdf(model_path)
    model, used = load_model_window([model_path], 0., 2.)
    np.testing.assert_allclose(model.wind_speed, 2.)
    assert used == [model_path]
    with pytest.raises(ValueError, match="Repeated model times"):
        load_model_window([model_path, model_path], 0., 2.)
    arco = ds.rename(sigma="lev", eastward_wind="uwind", northward_wind="vwind", solar_longitude="Ls")
    arco["MY_Ls"] = ("time", [34., 35., 35.])
    for name in ("temp", "tsurf", "psurf", "co2ice"):
        arco[name] = arco.uwind.copy()
        arco[name].attrs["units"] = dict(temp="K", tsurf="K", psurf="Pa", co2ice="kg m-2")[name]
    for name in ("uwind", "vwind"):
        arco[name].attrs["units"] = "m s-1"
    arco_path = tmp_path / "arco.nc"
    arco.to_netcdf(arco_path)
    selected, _ = load_arco_window([arco_path], 0., 2., {35})
    np.testing.assert_allclose(selected.uwind, -2.)
    np.testing.assert_allclose(selected.wind_speed, 2.)


def test_wind_height_comparison_uses_each_gas_constant_and_never_extrapolates():
    sigma = np.array([.1, .5, .95])
    temperature = xr.DataArray(np.full((3, 2, 2), 200.), dims=("sigma", "lat", "lon"),
                              coords=dict(sigma=sigma, lat=[-30., 30.], lon=[0., 180.]))
    for gas_constant in (188.92, 191.182):
        heights = gas_constant * 200. / 3.72076 * np.log(1. / sigma)
        eastward = xr.ones_like(temperature) * xr.DataArray(.01 * heights, dims="sigma", coords=dict(sigma=sigma))
        result = wind_at_height(temperature, eastward, -2. * eastward, gas_constant)
        np.testing.assert_allclose(result.eastward_wind, 10., rtol=1e-7)
        np.testing.assert_allclose(result.northward_wind, -20., rtol=1e-7)
        with pytest.raises(ValueError, match="no wind extrapolation"):
            wind_at_height(temperature, eastward, eastward, gas_constant, height_m=10.)
