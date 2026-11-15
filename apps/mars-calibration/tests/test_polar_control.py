"""A located cold cell must survive the map reduction and coordinate order."""
from pathlib import Path
import sys
import copy

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyse_polar_control import frost_deficit_map, validate_controls


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


@pytest.mark.parametrize("baseline", ["surface-only", "finite-rate"])
def test_phase_control_rejects_unmatched_source_or_extra_forcing_change(baseline):
    key = "atmospheric_condensation" if baseline == "surface-only" else "atmospheric_saturation_adjustment"
    first = dict(grid={"dt": 300}, forcing={}, input_sha256={"surface": "fixed"},
        co2_forcing=dict(atmospheric_condensation=baseline == "finite-rate",
            atmospheric_saturation_adjustment=baseline == "surface-only", latent_heat_j_kg=570000.))
    first["identity"] = dict(source_sha256={"physics.py": "fixed"}, co2_forcing=first["co2_forcing"])
    second = copy.deepcopy(first)
    second["co2_forcing"][key] = True
    validate_controls(first, second, baseline)
    second["identity"]["source_sha256"]["physics.py"] = "changed"
    with pytest.raises(ValueError, match="source"):
        validate_controls(first, second, baseline)
    second["identity"]["source_sha256"]["physics.py"] = "fixed"
    second["co2_forcing"]["latent_heat_j_kg"] = 600000.
    with pytest.raises(ValueError, match="Only"):
        validate_controls(first, second, baseline)
