"""Seasonal and local-time dust scenario history from the reference driver.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jnp


class LocalTimeState(NamedTuple):
    opacity: object
    bin_index: object


def validate_local_time_scenario(scenario,seasons,nodal_shape):
    """Check mapped fractions before tracing; source bins must span a full day."""
    if {'seasons','bins_hours','fractions'}-set(scenario):
        raise ValueError('local-time dust requires seasons, bins_hours and mapped fractions')
    axis=np.asarray(scenario['seasons']);bins=np.asarray(scenario['bins_hours'])
    if not np.array_equal(axis,np.asarray(seasons)):
        raise ValueError('local-time dust and opacity scenarios must share seasonal axes')
    if bins.ndim!=1 or len(bins)<2 or not np.all(np.isfinite(bins)) or np.any(np.diff(bins)<=0.) or bins[0]!=0. or bins[-1]!=24.:
        raise ValueError('local-time dust bins must increase from 0 to 24 hours')
    fractions=np.asarray(scenario['fractions'])
    if fractions.shape!=(len(axis),*nodal_shape,len(bins)-1) or not np.all(np.isfinite(fractions)):
        raise ValueError('local-time dust fractions must be finite and match seasons, grid and bins')


def seasonal_sample(seasons,values,solar_longitude_deg,*,segment=False):
    """Source seasonal interpolation and difference between bounding maps."""
    axis=jnp.asarray(seasons);maps=jnp.asarray(values);ls=jnp.mod(solar_longitude_deg,360.)
    upper=jnp.clip(jnp.searchsorted(axis,ls,side='right'),0,len(axis)-1)
    lower=jnp.mod(upper-1,len(axis))
    width=axis[upper]-axis[lower];width=jnp.where(width<0.,width+360.,width)
    weight=(axis[upper]-ls)/width
    value=maps[upper] if segment else weight*maps[lower]+(1.-weight)*maps[upper]
    return value,maps[upper]-maps[lower],lower,upper


def local_time_step(opacity_maps,fraction_maps,seasons,bins_hours,solar_longitude_deg,
                    longitude_rad,sols,state):
    """Original local-time bin update, including updating history after opacity.

    The axes share seasonal indices, as assumed by source initialization.
    Returned bin indices are zero-based; source comparisons are unchanged.
    """
    maps=jnp.asarray(opacity_maps)
    fraction,_,lower,upper=seasonal_sample(seasons,fraction_maps,solar_longitude_deg)
    local=jnp.mod(jnp.mod(sols+.5,1.)*24.-(180.-jnp.mod(longitude_rad*180./jnp.pi,360.))*12./180.,24.)
    index=jnp.searchsorted(jnp.asarray(bins_hours),local,side='right')-1
    index=jnp.broadcast_to(index,fraction.shape[:-1])
    current=jnp.take_along_axis(fraction,index[...,None],axis=-1)[...,0]
    if state is None:
        state=LocalTimeState(maps[upper],index)
    change=maps[upper]-maps[lower]
    opacity=jnp.where(change>0.,jnp.maximum(jnp.minimum(current*change+state.opacity,
        maps[upper]),maps[lower]),maps[upper])
    next_state=LocalTimeState(jnp.where(index!=state.bin_index,opacity,state.opacity),index)
    return opacity,current,next_state
