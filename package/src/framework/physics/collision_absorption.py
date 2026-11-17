"""Reference collision-induced infrared absorption by CO2/CO2 and CO2/H2.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.framework.gcm._dinosaur import jnp


@lru_cache(maxsize=1)
def load_tables():
    with np.load(Path(__file__).with_name('collision_absorption_tables.npz'),allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def interpolate_coefficients(table,temperature):
    """Original kinter: clamp 100--800 K and interpolate log(k) linearly."""
    t=jnp.clip(jnp.asarray(temperature),100.,800.)
    index=jnp.clip(jnp.searchsorted(jnp.arange(100.,801.,50.),t,side='left')-1,0,13)
    fraction=(t-(100.+50.*index))/50.
    values=jnp.asarray(table).T
    low,high=values[index],values[index+1]
    zero=(low==0.)&(high==0.)
    low=jnp.where(zero,1.,low);high=jnp.where(zero,1.,high)
    coefficient=jnp.exp(jnp.log(low)+fraction[...,None]*(jnp.log(high)-jnp.log(low)))
    return jnp.where(zero,0.,coefficient)


def collision_induced_depths(pressure_pa,temperature,*,co2_molar_fraction=1.,h2_molar_fraction=0.):
    """Original optci15 CIA on each subinterval, returning (subinterval,8,...).

    kinter mutates the source temperature before path-length/density evaluation.
    Preserve that 100--800 K clamp here. The first two ghost intervals have no
    CIA, regardless of whether the model top pressure is zero or positive.
    """
    p,t=map(jnp.asarray,(pressure_pa,temperature))
    if p.shape!=t.shape or p.shape[0]<5 or p.shape[0]%2!=1:
        raise ValueError('collision absorption requires matching 2*N+3 pressure/temperature grids')
    table=load_tables()
    mid_t=jnp.clip(.5*(t[1:]+t[:-1]),100.,800.)
    mid_p=.5*(p[1:]+p[:-1])/100.
    dp=jnp.diff(p,axis=0)/100.
    active=(jnp.arange(len(dp))>=2).reshape((len(dp),)+(1,)*(dp.ndim-1))
    safe_p=jnp.where(active,mid_p,1.)
    path=50.8*mid_t*dp/safe_p
    density=mid_p/mid_t*(273.15/1013.25)
    co2=co2_molar_fraction*density;h2=h2_molar_fraction*density
    carbon=interpolate_coefficients(table['co2_rotational'],mid_t)+interpolate_coefficients(table['co2_translational'],mid_t)
    hydrogen=interpolate_coefficients(table['co2_h2'],mid_t)
    depth=(co2[...,None]**2*carbon+h2[...,None]*co2[...,None]*hydrogen)*100.*path[...,None]
    return jnp.moveaxis(jnp.where(active[...,None],depth,0.),-1,1)
