import numpy as np
import pytest
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.thermal_relaxation import (
    damping_rate,fixed_tendency,height_tendency,analytic_tendency,load_equilibrium)


def test_timescale_units_disabled_forcing_and_temperature_gradient():
    np.testing.assert_allclose(damping_rate(jnp.array([-4.,345600.,0.])),[1./345600.,1./345600.,0.])
    temperature=jnp.array([170.,210.]);target=jnp.array([180.,190.])
    np.testing.assert_allclose(fixed_tendency(temperature,target,timescale=10.,mask=jnp.array([1.,0.])),[1.,0.])
    derivative=jax.grad(lambda x:jnp.sum(fixed_tendency(x,target,timescale=10.)))(temperature)
    np.testing.assert_allclose(derivative,[-.1,-.1])


def test_height_interpolation_preserves_target_under_mask_and_rejects_extrapolation():
    z=jnp.array([30.,20.,0.])[:,None];target=jnp.array([25.,10.,1.])[:,None]
    temperature=jnp.full((3,1),170.);equilibrium=jnp.array([200.,180.,160.])[:,None]
    tendency,teq=height_tendency(temperature,target,z,equilibrium,timescale=10.,mask=jnp.zeros((3,1)))
    np.testing.assert_allclose(teq[:,0],[190.,170.,161.]);np.testing.assert_array_equal(tendency,0.)
    invalid=height_tendency(temperature,target.at[0,0].set(31.),z,equilibrium)[0]
    assert np.isnan(invalid).all()
    derivative=jax.grad(lambda x:jnp.sum(height_tendency(temperature,x,z,equilibrium)[0]))(target)
    assert np.isfinite(derivative).all() and np.any(np.asarray(derivative)!=0.)


def test_analytic_boundary_rate_and_mask():
    pressure=jnp.array([.5,.8,1.,1.1])*1000.
    tendency,teq=analytic_tendency(jnp.full(4,300.),pressure,1000.,0.,1000.,kappa=.25,
        timescale=100.,surface_timescale=10.,temperature_zero=300.,stratosphere_temperature=300.,
        equator_pole_difference=0.,vertical_difference=0.,mask=jnp.array([1.,1.,0.,1.]))
    assert teq[2]==0. and tendency[2]==0.
    np.testing.assert_allclose(tendency[-1],-(300.-300.*1.1**.25)/100.,rtol=2e-6)
    with pytest.raises(ValueError):
        analytic_tendency(jnp.ones(4),pressure,1000.,0.,1000.,kappa=.25,sigma_boundary=1.)


@pytest.mark.parametrize('time_axis',[False,True])
def test_equilibrium_reader_keeps_equal_level_values_and_interpolates_other_counts(tmp_path,time_axis):
    xr=pytest.importorskip('xarray')
    from types import SimpleNamespace
    grid=SimpleNamespace(longitudes=np.deg2rad([0.,90.,359.]),latitudes=np.deg2rad([-80.,0.,80.]))
    values=np.broadcast_to(np.array([180.,220.,260.])[:,None,None],(3,2,4)).copy()
    path=tmp_path/'equilibrium.nc'
    def write(z):
        data=xr.Dataset({'teq':(('level','lat','lon'),values),'zzf':(('level','lat','lon'),
            np.broadcast_to(np.array(z)[:,None,None],values.shape)),
            'pfull':('level',[.1,1.,10.]),'lonb':('lonb',np.arange(0.,361.,90.)),
            'latb':('latb',[-90.,0.,90.])},coords={'lat':[-45.,45.],'lon':[45.,135.,225.,315.]})
        if time_axis: data=data.expand_dims(time=[0.])
        data.to_netcdf(path)
    write([30000.,15000.,0.])
    equal=load_equilibrium(grid,path,[0.,1.,2.,3.])
    np.testing.assert_allclose(equal,values[:,0,0,None,None]*np.ones((3,3,3)))
    mapped,z=load_equilibrium(grid,path,[0.,1.,2.,3.],height=True)
    np.testing.assert_allclose(mapped,equal);assert np.all(np.diff(z,axis=0)<0.)
    interpolated=load_equilibrium(grid,path,[0.,100.,1000.],spectral_core=True)
    np.testing.assert_allclose(interpolated[:,0,0],np.interp([50.,550.],[10.,100.,1000.],[180.,220.,260.]))
    with pytest.raises(ValueError,match='level counts'):
        load_equilibrium(grid,path,[0.,100.,1000.],height=True)
    values+=np.array([-10.,10.])[None,:,None]+np.array([0.,4.,8.,12.])[None,None,:]
    write([30000.,15000.,0.])
    centers=SimpleNamespace(longitudes=np.deg2rad([45.,135.,225.,315.]),latitudes=np.deg2rad([-45.,45.]))
    np.testing.assert_allclose(load_equilibrium(centers,path,[0.,1.,2.,3.]),values.transpose(0,2,1))
    write([30000.,0.,15000.])
    with pytest.raises(ValueError,match='decrease'):
        load_equilibrium(grid,path,[0.,1.,2.,3.],height=True)
