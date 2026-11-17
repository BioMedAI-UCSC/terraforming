"""Reference Bougher cool-to-space correction and solar non-LTE attenuation.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


@lru_cache(maxsize=1)
def reference_atmosphere():
    with np.load(Path(__file__).with_name('nonlte_reference.npz'),allow_pickle=False) as data:
        return {name:np.array(data[name]) for name in data.files if name!='source_git_blob'}


def interpolate_profile(log_pressure,reference_log_pressure,values):
    """Original interpnlte; out-of-range queries retain the last valid value."""
    x,y=map(jnp.asarray,(reference_log_pressure,values))
    def interpolate(previous,q):
        i=jnp.clip(jnp.searchsorted(x,q,side='right')-1,0,x.shape[0]-2)
        weight=(q-x[i])/(x[i+1]-x[i])
        current=y[i]*(1.-weight)+y[i+1]*weight
        current=jnp.where((q>=x[0])&(q<=x[-1]),current,previous)
        return current,current
    shape=jnp.shape(log_pressure)[1:]
    return jax.lax.scan(interpolate,jnp.broadcast_to(y[0],shape),log_pressure)[1]


def cooling(temperature,pressure_pa,co2_mass_fraction,oxygen_mass_fraction,*,reference=None,radiation_pressure_pa=None):
    """Original simple_nltecool at physical centers, returned in K/s.

    Preserve Fortran's integer 1/3 exponent: its temperature-exponent factor
    is identically zero. Preserve the pressure/sign gate and the source's
    density-ratio dependence on CO2 and atomic oxygen.
    """
    ref=reference_atmosphere() if reference is None else reference
    t,p,co2,oxygen=map(jnp.asarray,(temperature,pressure_pa,co2_mass_fraction,oxygen_mass_fraction))
    pmbar=p*.01
    if radiation_pressure_pa is None:
        query_pressure=p
    else:
        query_pressure=jnp.asarray(radiation_pressure_pa)
        if query_pressure.shape!=(2*t.shape[0]+3,*t.shape[1:]):
            raise ValueError('non-LTE radiation grid must interleave all physical layers')
    # Zero-mass ghost pressures use the source below-table interpolation value.
    logp=jnp.log10(jnp.where(query_pressure>0.,query_pressure*.01,1.e-300))
    refp=jnp.log10(jnp.asarray(ref['pressure_mbar']))
    def sample(values):
        result=interpolate_profile(logp,refp,values)
        return result if radiation_pressure_pa is None else result[3:-1:2]
    t0=sample(ref['temperature_k'])
    co20=10.**sample(jnp.log10(jnp.asarray(ref['co2_m3'])))
    o0=10.**sample(jnp.log10(jnp.asarray(ref['o_m3'])))
    q0=sample(ref['cooling_k_s'])
    def band(temp,c,o):
        ratio=1.7321e-13*jnp.sqrt(temp)/(5.2e-15*(1.+2.e-3*(temp-220.))*(temp/273.))
        density_ratio=(c+o)/(c+ratio*o)
        factor=.02/(pmbar*1.e3)*density_ratio
        return jnp.exp(-960./temp)/(1.+factor)
    result=q0*band(t,co2/.044,oxygen/.016)/band(t0,co20,o0)
    return jnp.where((pmbar>5.e-3)&(q0>=0.),0.,result)


def solar_heating_factor(pressure_pa):
    """Original solar non-LTE attenuation, using center pressure in Pa."""
    x=2.2e4*jnp.asarray(pressure_pa)*.01
    return x/(1.+x)


def infrared_blend(lte_heating,nonlte_heating,lookup_pressure_pa):
    """Original blend uses the upper-interface lookup pressure, not the center."""
    alpha=1./(1.+(jnp.asarray(lookup_pressure_pa)/.1)**4)
    return alpha*nonlte_heating+(1.-alpha)*lte_heating
