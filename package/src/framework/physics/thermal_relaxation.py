"""Released fixed, height-dependent and analytic thermal relaxation.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


def damping_rate(timescale):
    """Negative timescales use 86400-second days; positive values use seconds."""
    timescale=jnp.asarray(timescale)
    seconds=jnp.where(timescale<0.,-86400.*timescale,timescale)
    return jnp.where(seconds>0.,1./jnp.where(seconds>0.,seconds,1.),0.)


def fixed_tendency(temperature,equilibrium,*,timescale=-4.,mask=None):
    tendency=-damping_rate(timescale)*(jnp.asarray(temperature)-equilibrium)
    return tendency if mask is None else tendency*mask


def height_tendency(temperature,height,source_height,equilibrium,*,timescale=-4.,mask=None):
    """Interpolate each descending-height column; vertical dimension is first.

    As in FMS interp_1d, both grids must decrease strictly and target heights
    must be inside the source range. Invalid traced columns return NaNs instead
    of silently extending endpoint values.
    """
    temperature,height,source_height,equilibrium=map(jnp.asarray,
        (temperature,height,source_height,equilibrium))
    if temperature.shape!=height.shape or source_height.shape!=equilibrium.shape or \
            temperature.shape[1:]!=source_height.shape[1:] or source_height.shape[0]<2:
        raise ValueError('temperature and height columns must match; source needs at least two levels')
    source=source_height.reshape((source_height.shape[0],-1)).T
    target=height.reshape((height.shape[0],-1)).T
    values=equilibrium.reshape(source_height.shape[0],-1).T
    def interpolate(z,x,y):
        valid=jnp.all(jnp.diff(z)<0.)&jnp.all(jnp.diff(x)<0.)&(x[0]<=z[0])&(x[-1]>=z[-1])
        return jnp.where(valid,jnp.interp(-x,-z,y),jnp.nan)
    target_equilibrium=jax.vmap(interpolate)(source,target,values).T.reshape(height.shape)
    return fixed_tendency(temperature,target_equilibrium,timescale=timescale,mask=mask),target_equilibrium


def analytic_tendency(temperature,pressure,surface_pressure,latitude,reference_pressure,*,
                      kappa,timescale=-4.,surface_timescale=-4.,sigma_boundary=.7,
                      temperature_zero=315.,stratosphere_temperature=200.,
                      equator_pole_difference=60.,vertical_difference=10.,asymmetry=0.,mask=None):
    """Held--Suarez formula in the released routine, including its mask behavior."""
    if not np.isfinite(sigma_boundary) or sigma_boundary>=1.:
        raise ValueError('sigma boundary must be finite and below one')
    sin_lat=jnp.sin(latitude);cos_squared=1.-sin_lat*sin_lat
    normalized=jnp.asarray(pressure)/reference_pressure
    potential=(temperature_zero-equator_pole_difference*sin_lat*sin_lat-asymmetry*sin_lat-
        vertical_difference*cos_squared*jnp.log(normalized))
    equilibrium=jnp.maximum(potential*normalized**kappa,stratosphere_temperature-asymmetry*sin_lat)
    sigma=pressure/surface_pressure
    rate=damping_rate(timescale)
    coefficient=(damping_rate(surface_timescale)-rate)/(1.-sigma_boundary)
    rate=rate+jnp.where((sigma<=1.)&(sigma>sigma_boundary),cos_squared**2*coefficient*(sigma-sigma_boundary),0.)
    tendency=-rate*(temperature-equilibrium)
    if mask is not None:
        tendency=tendency*mask;equilibrium=equilibrium*mask
    return tendency,equilibrium


def load_equilibrium(grid,path,pressure_half,*,height=False,spectral_core=False):
    """Map released teq/zzf NetCDF inputs with the source's level-count rule.

    Pressure input pfull is in mbar. Equal source/model level counts bypass
    vertical interpolation, even when the pressures differ, as in read_teq.
    """
    import xarray as xr
    from src.celestials.planets.mars.dust_scenario import bilinear_map
    half=np.asarray(pressure_half,dtype=float)
    if half.ndim!=1 or len(half)<2 or not np.isfinite(half).all() or \
            half[0]<0. or np.any(np.diff(half)<=0.):
        raise ValueError('half-level pressures must be finite, nonnegative and increase')
    with xr.open_dataset(path,decode_times=False) as data:
        def sample(name):
            value=data[name]
            if 'time' in value.dims:
                if value.sizes['time']!=1: raise ValueError('equilibrium input requires one time sample')
                value=value.isel(time=0)
            return value
        def field(name):
            value=sample(name)
            vertical=[dimension for dimension in value.dims if dimension not in ('lat','lon')]
            if len(vertical)!=1: raise ValueError('equilibrium fields need one vertical dimension')
            result=np.asarray(value.transpose(vertical[0],'lat','lon'),dtype=float)
            if not np.isfinite(result).all(): raise ValueError('equilibrium fields must be finite')
            return result
        values=field('teq');levels=len(half)-1
        bounds=tuple(sample(name) for name in ('lonb','latb'))
        if np.any(values<=0.): raise ValueError('equilibrium temperature must be positive')
        if height:
            heights=field('zzf')
            if heights.shape!=values.shape or len(values)!=levels:
                raise ValueError('height equilibrium and model level counts must match the released reader')
            mapped=tuple(bilinear_map(grid,value,*bounds) for value in (values,heights))
            if np.any(np.diff(mapped[1],axis=0)>=0.):
                raise ValueError('equilibrium heights must decrease strictly')
            return mapped
        pressure=np.asarray(sample('pfull'),dtype=float)*100.
        if pressure.shape!=(len(values),) or not np.isfinite(pressure).all() or \
                np.any(pressure<=0.) or np.any(np.diff(pressure)<=0.):
            raise ValueError('equilibrium pressure levels must be positive and increase')
        if levels!=len(values):
            if len(values)<2: raise ValueError('pressure interpolation needs at least two source levels')
            if spectral_core:
                full=.5*(half[:-1]+half[1:])
            else:
                positive=half[:-1]>0.
                full=np.where(positive,np.diff(half)/np.log(half[1:]/np.where(positive,half[:-1],1.)),0.)
            values=np.apply_along_axis(lambda column:np.interp(full,pressure,column),0,values)
        return bilinear_map(grid,values,*bounds)
