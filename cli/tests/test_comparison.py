import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from cli.comparison import cap_export, load_source, metrics, normalize_units
from cli.main import cli


def test_gaussian_metrics_and_constant_reference():
    nodes, weights = np.polynomial.legendre.leggauss(4)
    lat = np.rad2deg(np.arcsin(nodes))
    reference = np.ones((4, 2))
    model = reference + np.arange(4)[:, None]
    result = metrics(model, reference, lat)
    assert result["bias"] == pytest.approx(np.dot(weights, np.arange(4)) / 2)
    assert result["rmse"] == pytest.approx(np.sqrt(np.dot(weights, np.arange(4)**2) / 2))
    assert result["correlation"] is None
    with pytest.raises(ValueError, match="Non-finite"):
        metrics(model * np.nan, reference, lat)


def test_units_and_mass_conversion():
    field = xr.DataArray([1.0], attrs={"units": "kg m-2"})
    assert float(normalize_units(field, "co2_ice")[0]) == pytest.approx(3.72076)
    with pytest.raises(ValueError, match="units"):
        normalize_units(xr.DataArray([1.0]), "temperature")


def test_explicit_selection_and_speed_before_average(tmp_path):
    coords = {"time": [0, 1, 2], "lev": [0.2, 0.9], "lat": [-30, 30], "lon": [0, 180]}
    winds = np.broadcast_to(np.array([2., -2., 100.])[:, None, None, None], (3, 2, 2, 2))
    ds = xr.Dataset({"uwind": (("time", "lev", "lat", "lon"), winds, {"units": "m s-1"}),
                     "vwind": (("time", "lev", "lat", "lon"), np.zeros_like(winds), {"units": "m s-1"}),
                     "Ls": ("time", [44., 46., 90.])}, coords=coords)
    ds.to_netcdf(tmp_path / "arco.nc")
    spec = {"name": "test", "paths": ["arco.nc"], "season_variable": "Ls", "mean_dims": ["time"], "isel": {"lev": -1}}
    loaded, records = load_source(spec, tmp_path, 45, 5)
    np.testing.assert_allclose(loaded.wind_speed, 2.)
    assert records[0]["selected_ls_range"] == [44., 46.]
    with pytest.raises(ValueError, match="dimensions"):
        load_source({**spec, "isel": {}}, tmp_path, 45, 5)
    with pytest.raises(ValueError, match="average time"):
        load_source({**spec, "mean_dims": []}, tmp_path, 45, 5)


def test_cap_export_preserves_two_dimensional_maps(tmp_path):
    ds = xr.Dataset({"temperature": (("lat", "lon"), [[200., 210.], [220., 230.]], {"units": "K"})}, coords={"lat": [-30, 30], "lon": [0, 180]})
    ds = ds.transpose("lon", "lat")
    terrain = xr.zeros_like(ds.temperature)
    cap_export(tmp_path, {"Model": ds, "MCD": ds}, terrain, {"temperature": (200, 230)})
    with xr.open_dataset(tmp_path / "source2/comparison.nc") as reopened:
        assert reopened.temperature.dims == ("lat", "lon")
        assert "time" not in reopened.dims
    template = (tmp_path / "comparison.in").read_text()
    assert "[comparison@1.temperature]-[comparison@2.temperature]" in template
    assert "Cmin, Cmax = 200,230" in template


def test_cli_exposes_compare_without_running_model():
    result = CliRunner().invoke(cli, ["mars", "compare", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
