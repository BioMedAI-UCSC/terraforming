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
