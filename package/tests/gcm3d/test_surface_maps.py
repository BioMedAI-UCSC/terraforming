from types import SimpleNamespace
import numpy as np
import pytest
import xarray as xr

pytest.importorskip('dinosaur')
from src.celestials.planets.mars.surface import (
    interpolate_surface_fields,reference_surface_fields_on_grid,conservative_surface_field)


def test_surface_interpolation_keeps_axes_hemispheres_and_longitude_seam():
    latitude=np.array([90.,0.,-90.])
    longitude=np.array([-180.,-90.,0.,90.])
    signal=latitude[:,None]+10.*np.cos(np.deg2rad(longitude))[None,:]
    dataset=xr.Dataset({'signal':(('lat','lon'),signal)},coords={'lat':latitude,'lon':longitude})
    grid=SimpleNamespace(latitudes=np.deg2rad([-45.,45.]),longitudes=np.deg2rad([0.,90.,359.]))
    actual=interpolate_surface_fields(grid,dataset)['signal']
    expected=np.array([10.,0.,10.*89./90.])[:,None]+np.array([-45.,45.])[None,:]
    assert actual.shape==(3,2)
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def write_reference_maps(path,grs=60.,emissivity=.95):
    for filename,variable,value in (
        ('mars_thermal_inertia8_2011.nc','thin',900.),
        ('mars_TES_albedo8_new.nc','albedo',.25),
        ('mars_sfc_emissivity8.nc','emis',emissivity),
        ('GRS_interp16.nc','grs_indices',grs),
    ):
        xr.Dataset({variable:(('lat','lon'),np.full((3,4),value))},
            coords={'lat':[-90.,0.,90.],'lon':[0.,90.,180.,270.]}).to_netcdf(path/filename)


def test_reference_maps_keep_native_northern_inertia_cap(tmp_path):
    write_reference_maps(tmp_path)
    grid=SimpleNamespace(latitudes=np.deg2rad([-80.,0.,65.,80.]),
        longitudes=np.deg2rad([0.,120.,240.]),quadrature_weights=np.ones(4))
    fields=reference_surface_fields_on_grid(grid,tmp_path)
    np.testing.assert_allclose(fields['thermal_inertia'],np.broadcast_to([900.,900.,900.,800.],(3,4)))
    for name,value in (('albedo',.25),('emissivity',.95),('grs_ice_latitude_deg',60.)):
        np.testing.assert_allclose(fields[name],value)


def test_reference_maps_reject_invalid_grs_instead_of_substituting_a_default(tmp_path):
    write_reference_maps(tmp_path,grs=120.)
    grid=SimpleNamespace(latitudes=np.array([0.]),longitudes=np.array([0.]),quadrature_weights=np.ones(1))
    with pytest.raises(ValueError,match='GRS'):
        reference_surface_fields_on_grid(grid,tmp_path)


def test_reference_emissivity_observation_is_preserved_without_clipping(tmp_path):
    write_reference_maps(tmp_path,emissivity=1.005)
    grid=SimpleNamespace(latitudes=np.array([0.]),longitudes=np.array([0.]),quadrature_weights=np.ones(1))
    fields=reference_surface_fields_on_grid(grid,tmp_path)
    np.testing.assert_allclose(fields['emissivity'],1.005)


def test_conservative_surface_remap_preserves_global_area_integral():
    from src.framework.gcm.coordinates import coordinate_system
    grid=coordinate_system('T21',4).horizontal
    values=np.random.default_rng(104).uniform(0.,1.,(4,8))
    field=xr.DataArray(values,dims=('lat','lon'))
    mapped=conservative_surface_field(grid,field)
    source_lat_area=np.diff(np.sin(np.linspace(-np.pi/2.,np.pi/2.,5)))
    expected=np.sum(values*source_lat_area[:,None])/(2.*8)
    quadrature=np.broadcast_to(np.asarray(grid.quadrature_weights),mapped.shape)
    actual=np.sum(mapped*quadrature)/quadrature.sum()
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)


def test_conservative_surface_remap_averages_both_sides_of_longitude_seam():
    grid=SimpleNamespace(latitudes=np.array([0.]),longitudes=np.deg2rad([0.,90.,180.,270.]),
        quadrature_weights=np.array([2.]))
    field=xr.DataArray([[2.,4.,6.,10.]],dims=('lat','lon'))
    actual=conservative_surface_field(grid,field)
    np.testing.assert_allclose(actual[:,0],[6.,3.,5.,8.],atol=1e-14)
