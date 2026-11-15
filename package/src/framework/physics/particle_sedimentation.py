"""Reference dust and coated-cloud moment sedimentation.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.co2_microphysics import sedimentation


class SedimentationResult(NamedTuple):
    tracers: object
    surface_deposit: object
    particle_density: object
    radius_m: object
    diagnostics: object


def particle_properties(tracers):
    """Original init_sedim for six physical moments, before any tagged copies."""
    q=jnp.maximum(jnp.asarray(tracers),1e-31)
    if q.shape[-1]!=6:
        raise ValueError('particle sedimentation requires six moment tracers')
    dust_radius=jnp.clip((q[...,0]/q[...,1]*.75/jnp.pi/2500.)**(1./3.)*
        jnp.exp(3.*.63676**2),1e-7,1e-5)
    mass=q[...,2]+q[...,4]
    density=jnp.clip((q[...,2]*917.+q[...,4]*2500.)/mass,917.,2500.)
    cloud_radius=jnp.clip((mass/q[...,3]*.75/jnp.pi/density)**(1./3.)*
        jnp.exp(3.*.3087**2),1e-7,1e-3)
    densities=jnp.stack([jnp.full_like(density,2500.),jnp.full_like(density,2500.),
        density,density,density],axis=-1)
    radii=jnp.stack([dust_radius,dust_radius*jnp.exp(-3.*.63676**2),
        cloud_radius,cloud_radius*jnp.exp(-3.*.3087**2),cloud_radius],axis=-1)
    return densities,radii


def sedimentation_step(tracers,pressure_half,pressure_full,temperature,temperature_surface,
                       heat_diffusivity,dt_seconds,*,scavenging=True,lifting_mask=None,
                       gas_constant=189.02,gravity=3.72):
    """Original init_sedim/sedim fluxes, deposition routing and lifting override.

    Tracers use the six-entry moment-water order. The shared bulk transport
    solves the same source continuity equation. Multiplying particle density
    in fallvel by exp(-stdv**2) applies the source moment fall-speed weight.
    Returned deposits follow source slots: dust+core in slot 0, ice in slot 5.
    Particle number lost at the ground is reported separately in diagnostics.
    """
    q,p,full,t,ts,k=map(jnp.asarray,(tracers,pressure_half,pressure_full,temperature,
        temperature_surface,heat_diffusivity))
    if q.shape!=(*t.shape,6):
        raise ValueError('moment tracers must match the sedimentation temperature grid')
    density,radius=particle_properties(q)
    q=jnp.maximum(q,0.)
    factor=jnp.exp(-jnp.array([.63676,.63676,.3087,.3087,.3087])**2)
    shape=radius.shape
    broadcast=lambda value:jnp.broadcast_to(value[...,None],shape)
    settled,deposit,coefficients=sedimentation(q[...,:5],
        jnp.broadcast_to(p[...,None],(t.shape[0]+1,*shape[1:])),broadcast(full),broadcast(t),
        jnp.broadcast_to(ts[...,None],shape[1:]),radius,
        jnp.broadcast_to(k[...,None],(t.shape[0]-1,*shape[1:])),dt_seconds,
        gas_constant=gas_constant,gravity=gravity,particle_density=density[1:]*factor)
    if not scavenging:
        settled=settled.at[...,4].set(q[...,4]);deposit=deposit.at[...,4].set(0.)
    new=q.at[...,:5].set(settled)
    routed=jnp.zeros_like(q[-1]).at[...,0].set(deposit[...,0]+deposit[...,4])
    routed=routed.at[...,5].set(deposit[...,2])
    if lifting_mask is not None:
        mask=jnp.asarray(lifting_mask,dtype=bool)
        if mask.shape!=q.shape[1:]:
            raise ValueError('lifting mask must have six entries per surface cell')
        new=new.at[-1].set(jnp.where(mask,q[-1],new[-1]))
        routed=jnp.where(mask,0.,routed)
    layer_mass=jnp.diff(p,axis=0)/gravity
    change=jnp.sum((new-q)*layer_mass[...,None],axis=0)
    return SedimentationResult(new,routed,density,radius,dict(transport=coefficients,
        per_species_deposit=deposit,atmospheric_change=change,
        dust_budget_residual=change[...,0]+change[...,4]+routed[...,0],
        water_budget_residual=change[...,2]+change[...,5]+routed[...,5]))
