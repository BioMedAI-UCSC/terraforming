"""Legacy top-to-bottom convective adjustment from the released Mars model.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from src.framework.gcm._dinosaur import jax,jnp


def adjust(temperature,u,v,tracers,pressure_half,pressure_full,kappa,*,gravity=3.72):
    """Original legacy_convect output fields, levels first and species last.

    The source's ghost/interfacial potential temperatures affect only pcon,
    which is never returned. Full-level heat and tracer updates below preserve
    the actual source loops, including continuing above a stable inner pair.
    Winds are unchanged by the original routine despite its mixing flag.
    """
    t,u,v,r,half,full=map(jnp.asarray,(temperature,u,v,tracers,pressure_half,pressure_full))
    if (len(t)<2 or u.shape!=t.shape or v.shape!=t.shape or full.shape!=t.shape or
        half.shape!=(len(t)+1,*t.shape[1:]) or r.shape[:-1]!=t.shape):
        raise ValueError('legacy convection requires matching atmosphere, pressure and tracer grids')
    # The source converts both pressures to mbar before taking this ratio.
    factor=((full/100.)/(half[-1]/100.))**kappa
    theta=t/factor
    mass=jnp.diff(half,axis=0)/gravity
    heat_weight=factor*mass
    levels=jnp.arange(len(t)).reshape((len(t),)+(1,)*(t.ndim-1))
    def layer(lower,fields):
        theta,r=fields
        unstable=theta[lower-1]<=theta[lower]
        def include(step,state):
            theta,r,mean,mean_q,heat_total,mass_total=state
            upper=jnp.maximum(lower-1-step,0)
            active=(step<lower)&unstable&(theta[upper]<=mean)
            combined=(theta[upper]*heat_weight[upper]+mean*heat_total)/(heat_weight[upper]+heat_total)
            combined_q=(r[upper]*mass[upper,...,None]+mean_q*mass_total[...,None])/(mass[upper]+mass_total)[...,None]
            mean=jnp.where(active&(theta[upper]!=mean),combined,mean)
            mean_q=jnp.where(active[...,None]&(r[upper]!=mean_q),combined_q,mean_q)
            selected=(levels>=upper)&(levels<=lower)&active[None]
            theta=jnp.where(selected,mean[None],theta)
            r=jnp.where(selected[...,None],mean_q[None],r)
            heat_total=heat_total+jnp.where(active,heat_weight[upper],0.)
            mass_total=mass_total+jnp.where(active,mass[upper],0.)
            return theta,r,mean,mean_q,heat_total,mass_total
        result=jax.lax.fori_loop(0,len(t)-1,include,
            (theta,r,theta[lower],r[lower],heat_weight[lower],mass[lower]))
        return result[:2]
    theta,r=jax.lax.fori_loop(1,len(t),layer,(theta,r))
    return theta*factor,u,v,r
