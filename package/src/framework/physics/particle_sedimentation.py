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


def dust_sedimentation_step(tracers,pressure_half,pressure_full,temperature,temperature_surface,
                            heat_diffusivity,dt_seconds,**options):
    """Use the native fixed-slot kernel with zero inactive cloud/core moments.

    State contains only mass/number pairs. Padding is local to the kernel call;
    no vapor, cloud, or water reservoir is introduced into the model state.
    """
    q=jnp.asarray(tracers)
    if q.shape[-1]<2 or q.shape[-1]%2:
        raise ValueError('dust sedimentation requires mass/number pairs')
    zero=jnp.zeros_like(q[...,0])
    fields=[q[...,0],q[...,1],zero,zero,zero,zero]
    for start in range(2,q.shape[-1],2): fields.extend([q[...,start],q[...,start+1],zero])
    packed=jnp.stack(fields,axis=-1)
    if options.get('lifting_mask') is not None:
        mask=jnp.asarray(options['lifting_mask'],dtype=bool)
        if mask.shape!=q.shape[1:]: raise ValueError('dust lifting mask must match surface mass/number pairs')
        off=jnp.zeros_like(mask[...,0]);fields=[mask[...,0],mask[...,1],off,off,off,off]
        for start in range(2,q.shape[-1],2): fields.extend([mask[...,start],mask[...,start+1],off])
        options['lifting_mask']=jnp.stack(fields,axis=-1)
    result=sedimentation_step(packed,pressure_half,pressure_full,temperature,temperature_surface,
        heat_diffusivity,dt_seconds,**options)
    indices=jnp.asarray([0,1]+[index for start in range(6,packed.shape[-1],3) for index in (start,start+1)])
    deposits=jnp.asarray([0]+list(range(6,packed.shape[-1],3)))
    particle_indices=jnp.asarray([0,1]+[index for start in range(5,result.radius_m.shape[-1],3) for index in (start,start+1)])
    diagnostics=dict(result.diagnostics)
    diagnostics['atmospheric_change']=result.diagnostics['atmospheric_change'][...,indices]
    diagnostics['per_species_deposit']=result.diagnostics['per_species_deposit'][...,particle_indices]
    return SedimentationResult(result.tracers[...,indices],result.surface_deposit[...,deposits],
        result.particle_density[...,particle_indices],result.radius_m[...,particle_indices],diagnostics)


def particle_properties(tracers):
    """Original init_sedim: six cloud moments, then extra dust mass/number/core."""
    q=jnp.maximum(jnp.asarray(tracers),1e-31)
    if q.shape[-1]<6 or (q.shape[-1]-6)%3:
        raise ValueError('particle sedimentation requires six moments and dust mass/number/core triples')
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
    for start in range(6,q.shape[-1],3):
        radius=jnp.clip((q[...,start]/q[...,start+1]*.75/jnp.pi/2500.)**(1./3.)*
            jnp.exp(3.*.63676**2),1e-7,1e-5)
        densities=jnp.concatenate([densities,jnp.stack([jnp.full_like(density,2500.),
            jnp.full_like(density,2500.),density],axis=-1)],axis=-1)
        radii=jnp.concatenate([radii,jnp.stack([radius,radius*jnp.exp(-3.*.63676**2),
            cloud_radius],axis=-1)],axis=-1)
    return densities,radii


def sedimentation_step(tracers,pressure_half,pressure_full,temperature,temperature_surface,
                       heat_diffusivity,dt_seconds,*,scavenging=True,lifting_mask=None,
                       gas_constant=189.02,gravity=3.72):
    """Original init_sedim/sedim fluxes, deposition routing and lifting override.

    Tracers use the six-entry moment-water order followed by physical dust
    mass/number/core triples. The shared bulk transport
    solves the same source continuity equation. Multiplying particle density
    in fallvel by exp(-stdv**2) applies the source moment fall-speed weight.
    Returned deposits follow source slots: dust+core in slot 0, ice in slot 5.
    Particle number lost at the ground is reported separately in diagnostics.
    """
    q,p,full,t,ts,k=map(jnp.asarray,(tracers,pressure_half,pressure_full,temperature,
        temperature_surface,heat_diffusivity))
    if q.shape[:-1]!=t.shape:
        raise ValueError('moment tracers must match the sedimentation temperature grid')
    density,radius=particle_properties(q)
    q=jnp.maximum(q,0.)
    extra_modes=(q.shape[-1]-6)//3
    indices=jnp.asarray([0,1,2,3,4]+list(range(6,q.shape[-1])))
    factor=jnp.exp(-jnp.array([.63676,.63676,.3087,.3087,.3087]+
        [.63676,.63676,.3087]*extra_modes)**2)
    shape=radius.shape
    broadcast=lambda value:jnp.broadcast_to(value[...,None],shape)
    settled,deposit,coefficients=sedimentation(q[...,indices],
        jnp.broadcast_to(p[...,None],(t.shape[0]+1,*shape[1:])),broadcast(full),broadcast(t),
        jnp.broadcast_to(ts[...,None],shape[1:]),radius,
        jnp.broadcast_to(k[...,None],(t.shape[0]-1,*shape[1:])),dt_seconds,
        gas_constant=gas_constant,gravity=gravity,particle_density=density[1:]*factor)
    if not scavenging:
        cores=jnp.asarray([4]+list(range(7,5+3*extra_modes,3)))
        settled=settled.at[...,cores].set(q[...,indices[cores]])
        deposit=deposit.at[...,cores].set(0.)
    new=q.at[...,indices].set(settled)
    routed=jnp.zeros_like(q[-1]).at[...,0].set(deposit[...,0]+deposit[...,4])
    routed=routed.at[...,5].set(deposit[...,2])
    for start in range(6,q.shape[-1],3):
        routed=routed.at[...,start].set(deposit[...,start-1]+deposit[...,start+1])
    if lifting_mask is not None:
        mask=jnp.asarray(lifting_mask,dtype=bool)
        if mask.shape!=q.shape[1:]:
            raise ValueError('lifting mask must have one entry per surface tracer')
        new=new.at[-1].set(jnp.where(mask,q[-1],new[-1]))
        routed=jnp.where(mask,0.,routed)
    layer_mass=jnp.diff(p,axis=0)/gravity
    change=jnp.sum((new-q)*layer_mass[...,None],axis=0)
    dust_indices=jnp.asarray([0,4]+[i for start in range(6,q.shape[-1],3) for i in (start,start+2)])
    mass_indices=jnp.asarray([0]+list(range(6,q.shape[-1],3)))
    return SedimentationResult(new,routed,density,radius,dict(transport=coefficients,
        per_species_deposit=deposit,atmospheric_change=change,
        dust_budget_residual=jnp.sum(change[...,dust_indices],axis=-1)+jnp.sum(routed[...,mass_indices],axis=-1),
        water_budget_residual=change[...,2]+change[...,5]+routed[...,5]))
