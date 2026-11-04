"""CI checks of the persistent result contract without running a GCM."""
import json
import numpy as np
import pytest
pytest.importorskip('h5netcdf')
import xarray as xr
from cli.aerosol_results import write_results


def frame(temperature,burden):
    return dict(lat=[-30.,30.],lon=[0.,120.,240.],maps=dict(
        surface_temperature=dict(label='Surface temperature',units='K',data=np.full((2,3),temperature).tolist()),
        aerosol_burden=dict(label='Suspended aerosol',units='kg/m²',data=np.full((2,3),burden).tolist())))


def test_result_bundle_retains_control_and_measured_difference(tmp_path):
    report=dict(status='completed',elapsed_seconds=3600.,cases=dict(
        off=dict(surface_temperature_k=210.),active=dict(surface_temperature_k=210.5)))
    summary=write_results(tmp_path,report,dict(off=frame(210.,0.),active=frame(210.5,2e-6)),{'particle':'source-backed rod'})
    assert summary['final_area_mean_surface_difference_k']==.5
    assert json.loads((tmp_path/'summary.json').read_text())['status']=='completed'
    assert json.loads((tmp_path/'fields.json').read_text())['maps']['surface_temperature']['units']=='K'
    with xr.open_dataset(tmp_path/'results.nc',engine='h5netcdf') as data:
        np.testing.assert_allclose(data.surface_temperature_active_minus_off,.5)
        np.testing.assert_allclose(data.off_aerosol_burden,0.)
        np.testing.assert_allclose(data.aerosol_burden,2e-6)
        assert data.attrs['primary_case']=='active'
        assert 'unverified' in data.attrs['interpretation']


def test_comparison_rejects_different_grids(tmp_path):
    changed=frame(210.,0.);changed['lat']=[-45.,45.]
    with pytest.raises(ValueError,match='grids differ'):
        write_results(tmp_path,dict(status='completed',elapsed_seconds=1.),
            dict(off=changed,active=frame(211.,2e-6)),{})
