from types import SimpleNamespace
import numpy as np
import pytest

from src.celestials.planets.mars.dust_scenario import bilinear_map,load_scenario_on_grid


def grid(lon,lat):
    return SimpleNamespace(longitudes=np.deg2rad(lon),latitudes=np.deg2rad(lat))


def test_bilinear_dust_map_extends_polar_rows_and_keeps_longitude_orientation():
    values=np.array([[1.,2.,3.,4.],[10.,20.,30.,40.]])
    actual=bilinear_map(grid([45.,135.,225.,315.],[-90.,-45.,0.,45.,90.]),
        values,[0.,90.,180.,270.,360.],[-90.,0.,90.])
    np.testing.assert_allclose(actual[:,0],values[0],rtol=2e-15)
    np.testing.assert_allclose(actual[:,-1],values[1],rtol=2e-15)
    np.testing.assert_allclose(actual[:,2],.5*(values[0]+values[1]),rtol=2e-15)


def test_bilinear_dust_map_uses_periodic_seam_and_retains_leading_season_axis():
    values=np.array([[1.,2.,3.,4.],[10.,20.,30.,40.]])
    maps=np.stack([values,2.*values])
    actual=bilinear_map(grid([-.1,0.,359.9],[-45.,45.]),maps,
        [0.,90.,180.,270.,360.],[-90.,0.,90.])
    assert actual.shape==(2,3,2)
    np.testing.assert_allclose(actual[0,0],actual[0,2],rtol=1e-14)
    np.testing.assert_allclose(actual[1],2.*actual[0],rtol=1e-14)
    np.testing.assert_allclose(actual[0,1],[2.5,25.],rtol=1e-9)


def test_dust_scenario_rejects_an_unpinned_file_before_reading_fields(tmp_path):
    path=tmp_path/'DustScenario_Background.nc';path.write_bytes(b'bad data')
    with pytest.raises(ValueError,match='pinned reference'):
        load_scenario_on_grid(grid([0.],[0.]),path)


def test_dust_map_rejects_nonfinite_and_misaligned_fields():
    for values in (np.zeros((2,3)),np.full((2,4),np.nan)):
        with pytest.raises(ValueError):
            bilinear_map(grid([0.],[0.]),values,[0.,90.,180.,270.,360.],[-90.,0.,90.])


def test_injected_opacity_maps_keep_mode_order_seasons_and_polar_rows(tmp_path):
    import xarray as xr
    from src.celestials.planets.mars.dust_scenario import load_injection_fractions_on_grid
    seasons=np.array([0.,120.,240.,360.])
    values=np.arange(32,dtype=float).reshape(4,2,4)/100.
    dataset=xr.Dataset({'areo':('time',seasons),'bins':('mode',[3,1]),
        'rad03':(('time','lat','lon'),values),'rad01':(('time','lat','lon'),values+.5),
        'lonb':('lonb',[0.,90.,180.,270.,360.]),'latb':('latb',[-90.,0.,90.])})
    path=tmp_path/'radius_scenario.nc';dataset.to_netcdf(path)
    target=grid([45.,135.,225.,315.],[-90.,-45.,45.,90.])
    actual=load_injection_fractions_on_grid(target,path,2)
    expected=np.stack([values[:,0],values[:,0],values[:,1],values[:,1]],axis=-1)
    np.testing.assert_allclose(actual['fractions'][...,0],expected,rtol=2e-15,atol=1e-15)
    np.testing.assert_allclose(actual['fractions'][...,1],expected+.5,rtol=2e-15)
    np.testing.assert_array_equal(actual['seasons'],seasons)
    with pytest.raises(ValueError,match='physical mode'): load_injection_fractions_on_grid(target,path,1)
    for changed in (dataset.assign(areo=('time',[0.,120.,120.,360.])),
                    dataset.assign(areo=('time',[0.,120.,240.,359.])),
                    dataset.assign(rad03=dataset.rad03-1.),
                    dataset.assign(rad03=dataset.rad03*np.nan)):
        changed.to_netcdf(path)
        with pytest.raises(ValueError): load_injection_fractions_on_grid(target,path,2)


def test_local_time_file_uses_hour_labels_and_preserves_seasonal_fields(tmp_path):
    import xarray as xr
    from src.celestials.planets.mars.dust_scenario import load_local_time_fractions_on_grid
    values=np.arange(16,dtype=float).reshape(2,2,4)/100.
    dataset=xr.Dataset({'areo':('time',[0.,360.]),'bins':('bins',[0,12,24]),
        'ltfrac00':(('time','lat','lon'),values),'ltfrac12':(('time','lat','lon'),values+.5),
        'lonb':('lonb',[0.,90.,180.,270.,360.]),'latb':('latb',[-90.,0.,90.])})
    path=tmp_path/'localtime_frac.nc';dataset.to_netcdf(path)
    actual=load_local_time_fractions_on_grid(grid([45.,135.,225.,315.],[-45.,45.]),path)
    np.testing.assert_array_equal(actual['bins_hours'],[0.,12.,24.])
    np.testing.assert_allclose(actual['fractions'][...,0],values.transpose(0,2,1),rtol=2e-15,atol=1e-15)
    np.testing.assert_allclose(actual['fractions'][...,1],values.transpose(0,2,1)+.5,rtol=2e-15)
    for bins in ([0,12,23],[0,0,24],[0.,12.5,24.]):
        dataset.assign(bins=('bins',bins)).to_netcdf(path)
        with pytest.raises(ValueError,match='local-time bins'):
            load_local_time_fractions_on_grid(grid([45.],[-45.]),path)
