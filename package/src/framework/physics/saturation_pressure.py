"""Pinned FMS saturation tables and humidity equations in differentiable JAX.

SPDX-License-Identifier: LGPL-3.0-only
Derived from NOAA-GFDL FMS revision 71957e9d; NASA's Apache-2.0 patch supplies
the extended table bounds. See licenses/SATURATION-PRESSURE-NOTICE.txt.
"""
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


class VaporPressure(NamedTuple):
    pressure_pa: object
    derivative_pa_k: object
    valid: object


class Saturation(NamedTuple):
    specific_humidity: object
    temperature_derivative: object
    vapor_pressure_pa: object
    valid: object


@lru_cache(maxsize=1)
def load_tables():
    with np.load(Path(__file__).with_name('saturation_tables.npz'),allow_pickle=False) as data:
        return {name:data[name].copy() for name in data.files}


def vapor_pressure(temperature,*,model='smithsonian',phase='blend'):
    """Original nearest-node quadratic lookup and supplied d(es)/dT.

    Phase is the source ice/water blend, liquid only, or a freezing-point
    ice/liquid switch. Invalid table indices return NaNs and valid=False;
    the original FMS wrapper raises a fatal error for them.
    """
    if model not in ('smithsonian','simple'):
        raise ValueError('saturation model must be smithsonian or simple')
    if phase not in ('blend','liquid','ice-liquid'):
        raise ValueError('saturation phase must be blend, liquid or ice-liquid')
    data=load_tables();minimum=float(data[model+'_minimum_k'])
    table=jnp.asarray(data[model+'_tables'][('blend','liquid','ice-liquid').index(phase)])
    t=jnp.asarray(temperature)
    delta_t=jax.lax.optimization_barrier(t-minimum)
    shifted=jax.lax.optimization_barrier(delta_t+.05)
    index=jnp.trunc(10.*shifted).astype(int)
    valid=jnp.isfinite(t)&(index>=0)&(index<table.shape[-1])
    safe=jnp.clip(index,0,table.shape[-1]-1)
    delta=jnp.where(valid,delta_t,0.)-.1*safe
    es=table[0,safe]+delta*(table[1,safe]+delta*table[2,safe])
    derivative=table[1,safe]+2.*delta*table[2,safe]
    return VaporPressure(jnp.where(valid,es,jnp.nan),jnp.where(valid,derivative,jnp.nan),valid)


def specific_humidity(temperature,pressure_pa,*,relative_humidity=1.,vapor_ratio=None,
                      use_exact_qs=False,gas_constant=189.02,vapor_gas_constant=461.50,
                      model='smithsonian',phase='blend'):
    """Original compute_qs, including its nonpositive-denominator behavior.

    Keep the explicit source derivative even when qs is capped at epsilon.
    With vapor_ratio supplied and use_exact_qs=True, use its alternate exact
    expression. No derivative is detached from the JAX computation.
    """
    vapor=vapor_pressure(temperature,model=model,phase=phase)
    p=jnp.asarray(pressure_pa);epsilon=gas_constant/vapor_gas_constant
    es=vapor.pressure_pa*relative_humidity;des=vapor.derivative_pa_k*relative_humidity
    if vapor_ratio is not None and use_exact_qs:
        factor=1.+(vapor_gas_constant/gas_constant-1.)*jnp.asarray(vapor_ratio)
        qs=factor*epsilon*es/p;derivative=factor*epsilon*des/p
    else:
        denominator=p-(1.-epsilon)*es
        qs=jnp.where(denominator>0.,epsilon*es/jnp.where(denominator>0.,denominator,1.),epsilon)
        safe=jnp.where(denominator!=0.,denominator,1.)
        derivative=jnp.where(denominator!=0.,epsilon*p*des/safe**2,jnp.inf)
    valid=vapor.valid&jnp.isfinite(p)&(p>0.)
    return Saturation(jnp.where(valid,qs,jnp.nan),jnp.where(valid,derivative,jnp.nan),
        jnp.where(valid,es,jnp.nan),valid)
