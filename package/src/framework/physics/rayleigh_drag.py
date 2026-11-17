"""Reference lower-atmosphere, upper-atmosphere and tropical Rayleigh drag.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from src.framework.gcm._dinosaur import jnp


def rayleigh_tendencies(u,v,pressure_full,pressure_surface,latitude_rad,*,
                        timescale=-4000.e6,boundary_sigma=.7,
                        model_top=False,top_tau_days=1.,top_inflex_pa=.1,top_sigma_cutoff=1.,
                        tropical=False,tropical_tau_days=1.,tropical_inflex_pa=.1,
                        tropical_sigma_cutoff=1.,tropical_latitude_width_deg=30.,mask=None):
    """Original rayleigh_damping on level-first arrays, returning m/s2.

    A negative timescale specifies days of 86400 seconds, as in the source;
    a positive value specifies seconds and zero disables surface drag. The
    source compares its nominal sponge cutoffs with sigma, despite the pressure
    names in its namelist. No wind clipping is applied.
    """
    u,v,p,ps,latitude=map(jnp.asarray,(u,v,pressure_full,pressure_surface,latitude_rad))
    if u.shape!=v.shape or p.shape!=u.shape:
        raise ValueError('Rayleigh winds and full-level pressure must have matching shapes')
    sigma=p/ps
    tau=jnp.asarray(timescale)
    tau=jnp.where(tau<0.,-86400.*tau,tau)
    frequency=jnp.where(tau>0.,1./jnp.where(tau>0.,tau,1.),0.)
    rate=jnp.where((sigma<=1.)&(sigma>boundary_sigma),
        frequency*(sigma-boundary_sigma)/(1.-boundary_sigma),0.)
    if model_top:
        top=.5*(1.+jnp.tanh(1.5*jnp.log(top_inflex_pa/p)))/(top_tau_days*86400.)
        rate=rate+jnp.where(sigma<top_sigma_cutoff,top,0.)
    if tropical:
        structure=jnp.exp(-(latitude*(180./jnp.pi)/tropical_latitude_width_deg)**4)
        trop=.5*structure*(1.+jnp.tanh(.8*jnp.log(tropical_inflex_pa/p)))/(
            tropical_tau_days*86400.)
        rate=rate+jnp.where(sigma<tropical_sigma_cutoff,trop,0.)
    if mask is not None:
        rate=rate*jnp.asarray(mask)
    return -rate*u,-rate*v
