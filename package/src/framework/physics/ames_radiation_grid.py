"""Ames fillpt radiation subgrid, with vertical levels on the first axis.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp


class RadiationGrid(NamedTuple):
    pressure: object
    temperature: object
    lookup_pressure: object
    lookup_temperature: object
    delta_pressure: object


def radiation_grid(pressure_half,pressure_full,temperature,surface_temperature,
                   stratosphere_temperature=None):
    """Interleave physical centers/interfaces and apply the original fillpt.

    Pressures must be positive, increasing, and use the same units. The top
    interface is supplied explicitly; this routine does not invent a pressure
    floor for Dinosaur's zero-pressure top. Invalid columns return NaNs.
    """
    half,full,t=map(jnp.asarray,(pressure_half,pressure_full,temperature))
    if half.shape!=(t.shape[0]+1,)+t.shape[1:] or full.shape!=t.shape:
        raise ValueError("radiation pressure arrays must match the atmospheric column")
    top=t[0] if stratosphere_temperature is None else jnp.asarray(stratosphere_temperature)
    n=t.shape[0]
    p=jnp.zeros((2*n+3,)+t.shape[1:],dtype=jnp.result_type(half,full,float))
    p=p.at[0].set(half[0]*.5e-5).at[1].set(half[0]*.5)
    p=p.at[2::2].set(half).at[3::2].set(full)
    valid=jnp.all((half[:-1]>0.)&(full>half[:-1])&(full<half[1:]),axis=0)
    # Inactive invalid columns must not introduce log(0) into reverse mode.
    safe=jnp.where(valid[None],p,jnp.ones_like(p))
    tl=jnp.zeros_like(p).at[:3].set(top).at[3::2].set(t)
    numerator=jnp.log(safe[4:-1:2]/safe[5::2])
    denominator=jnp.log(safe[3:-2:2]/safe[5::2])
    denominator=jnp.where(valid[None],denominator,1.)
    interface=t[1:]+(t[:-1]-t[1:])*numerator/denominator
    tl=tl.at[4:-1:2].set(interface).at[-1].set(surface_temperature)
    lookup_t=jnp.concatenate([jnp.broadcast_to(top,(2,)+t.shape[1:]),
        jnp.repeat(tl[2:-1:2],2,axis=0),tl[-1:]],axis=0)
    lookup_p=jnp.concatenate([jnp.broadcast_to(half[0]*.5,(2,)+t.shape[1:]),
        jnp.repeat(half[:-1],2,axis=0),half[-1:]],axis=0)
    values=(p,tl,lookup_p,lookup_t,jnp.diff(half,axis=0))
    return RadiationGrid(*(jnp.where(valid[None],v,jnp.nan) for v in values))
