"""Local conservative transport for grid-space reference tracers.

Winds and pressure remain spectral. Physical tracers stay on the Gaussian grid
so microphysics does not acquire remote Gibbs lobes on every time step.
"""
import dataclasses
import math
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp,primitive_equations,spherical_harmonic


def tracer_to_nodal(grid,value):
    """Read both grid-space reference tracers and older spectral tracers."""
    if value.shape[-2:]==grid.nodal_shape:
        return value
    if value.shape[-2:]!=grid.modal_shape:
        raise ValueError('tracer shape must match the nodal or modal grid')
    return grid.to_nodal(value)


def tracer_from_nodal(grid,value,like):
    """Retain the caller's tracer representation when updating shared physics."""
    if like.shape[-2:]==grid.nodal_shape:
        return value
    if like.shape[-2:]!=grid.modal_shape:
        raise ValueError('tracer shape must match the nodal or modal grid')
    return grid.to_modal(value)


def project_tracers(previous,current,coords,*,conserve_previous=True):
    """Close roundoff inventory errors without mixing distant cells or layers.

    Only roundoff-size negative values can be removed. A material negative
    tracer is invalid: global blending must not hide a failed transport step.
    A scalar inventory correction preserves all zeros and spatial structure.
    """
    if set(previous.tracers)!=set(current.tracers):
        raise ValueError('transport projection requires the same tracer names before and after the step')
    grid=coords.horizontal
    ds=jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]
    carrier=jnp.exp(grid.to_nodal(current.log_surface_pressure))*ds*grid.quadrature_weights
    old_carrier=jnp.exp(grid.to_nodal(previous.log_surface_pressure))*ds*grid.quadrature_weights
    tracers={};diagnostics={}
    gas_pair={'co2_mmr_gas','ar_mmr_gas'}<=set(current.tracers)
    for name,value in current.tracers.items():
        nodal=tracer_to_nodal(grid,value)
        raw=jnp.sum(nodal*carrier)
        target=(jnp.sum(tracer_to_nodal(grid,previous.tracers[name])*old_carrier)
            if conserve_previous and not (gas_pair and name=='co2_mmr_gas') else raw)
        margin=32.*jnp.finfo(nodal.dtype).eps*jnp.max(jnp.abs(nodal))
        positive=jnp.maximum(nodal,0.)
        actual=jnp.sum(positive*carrier)
        result=positive*(target/jnp.where(actual>0.,actual,1.))
        valid=(jnp.min(nodal)>=-margin)&(target>=0.)&((actual>0.)|(target==0.))&jnp.isfinite(target)
        tracers[name]=jnp.where(valid,result,jnp.nan)
        diagnostics[name]=dict(raw_inventory=raw,target_inventory=target,
            corrected_inventory=jnp.sum(tracers[name]*carrier),raw_inventory_change=raw-target,
            redistributed_inventory=jnp.sum(jnp.abs(tracers[name]-nodal)*carrier),
            minimum_before=jnp.min(nodal),minimum_after=jnp.min(tracers[name]),valid=valid)
    return dataclasses.replace(current,tracers=tracers),diagnostics


def transport_tracers(previous,current,coords,dt):
    """Donor-cell fluxes on Gaussian cells, with CFL-controlled substeps.

    ``dt`` and winds use Dinosaur's nondimensional units. Longitude is periodic;
    latitude and sigma have closed polar/top/bottom boundaries. The maximum
    substep count is the next power of two above one longitude ring; larger CFLs return an invalid state.
    Static scan branches retain forward and reverse JAX derivatives.
    This is a first-order transport adaptation, not FV3's higher-order limiter.
    """
    if not previous.tracers:
        return current,{}
    grid=coords.horizontal
    names=tuple(previous.tracers)
    ds=np.diff(coords.vertical.boundaries)[:,None,None]
    weights=np.broadcast_to(np.asarray(grid.quadrature_weights),grid.nodal_shape)[0]
    sin_edges=np.r_[-1.,-1.+np.cumsum(weights)*2./weights.sum()]
    sin_edges[-1]=1.
    edges=np.arcsin(np.clip(sin_edges,-1.,1.))
    dlon=2.*np.pi/grid.nodal_shape[0]
    area=jnp.asarray(dlon*np.diff(sin_edges))[None,:]
    volume=jnp.asarray(ds)*area
    before=dataclasses.replace(previous,tracers={})
    after=dataclasses.replace(current,tracers={})
    old=primitive_equations.compute_diagnostic_state_sigma(before,coords)
    new=primitive_equations.compute_diagnostic_state_sigma(after,coords)
    u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,
        .5*(previous.vorticity+current.vorticity),.5*(previous.divergence+current.divergence))
    east=.5*(u+jnp.roll(u,-1,axis=1))*jnp.asarray(ds)*jnp.asarray(np.diff(edges))[None,None,:]/grid.radius
    north=.5*(v[:,:,:-1]+v[:,:,1:])*jnp.asarray(ds)*jnp.asarray(np.cos(edges[1:-1]))[None,None,:]*dlon/grid.radius
    down=.5*(old.sigma_dot_full+new.sigma_dot_full)*area
    zeros_lat=jnp.zeros_like(v[:,:,:1]);zeros_level=jnp.zeros_like(v[:1])
    outgoing=jnp.maximum(east,0.)+jnp.maximum(-jnp.roll(east,1,axis=1),0.)
    outgoing+=jnp.concatenate([jnp.maximum(north,0.),zeros_lat],axis=2)
    outgoing+=jnp.concatenate([zeros_lat,jnp.maximum(-north,0.)],axis=2)
    outgoing+=jnp.concatenate([jnp.maximum(down,0.),zeros_level],axis=0)
    outgoing+=jnp.concatenate([zeros_level,jnp.maximum(-down,0.)],axis=0)
    cfl=dt*jnp.max(outgoing/volume)
    maximum=2**math.ceil(math.log2(grid.nodal_shape[0]))
    valid=jnp.isfinite(cfl)&(cfl<=.8*maximum)
    required=jnp.maximum(jnp.where(jnp.isfinite(cfl),cfl/.8,maximum),1.)
    branch=jnp.clip(jnp.ceil(jnp.log2(required)).astype(jnp.int32),0,int(math.log2(maximum)))
    carrier=jnp.exp(grid.to_nodal(previous.log_surface_pressure))*volume
    mass=jnp.stack([tracer_to_nodal(grid,previous.tracers[name])*carrier for name in names],axis=-1)

    def integrate(count):
        def advance(value,_):
            density=value/volume[...,None]
            east_flux=jnp.maximum(east,0.)[...,None]*density+jnp.minimum(east,0.)[...,None]*jnp.roll(density,-1,axis=1)
            north_flux=jnp.maximum(north,0.)[...,None]*density[:,:,:-1]+jnp.minimum(north,0.)[...,None]*density[:,:,1:]
            down_flux=jnp.maximum(down,0.)[...,None]*density[:-1]+jnp.minimum(down,0.)[...,None]*density[1:]
            north_flux=jnp.pad(north_flux,((0,0),(0,0),(1,1),(0,0)))
            down_flux=jnp.pad(down_flux,((1,1),(0,0),(0,0),(0,0)))
            tendency=jnp.roll(east_flux,1,axis=1)-east_flux
            tendency+=north_flux[:,:,:-1]-north_flux[:,:,1:]
            tendency+=down_flux[:-1]-down_flux[1:]
            return value+(dt/count)*tendency,None
        return lambda value:jax.lax.scan(jax.checkpoint(advance),value,None,length=count)[0]

    mass=jax.lax.switch(branch,[integrate(2**i) for i in range(int(math.log2(maximum))+1)],mass)
    new_carrier=jnp.exp(grid.to_nodal(current.log_surface_pressure))*volume
    tracers={name:jnp.where(valid,mass[...,i]/new_carrier,jnp.nan) for i,name in enumerate(names)}
    if {'co2_mmr_gas','ar_mmr_gas'}<=set(tracers):
        # Assign the dry core's local carrier discrepancy to its condensible gas.
        tracers['co2_mmr_gas']=1.-tracers['ar_mmr_gas']
    transported,diagnostics=project_tracers(previous,dataclasses.replace(current,tracers=tracers),coords)
    if {'co2_mmr_gas','ar_mmr_gas'}<=set(transported.tracers):
        transported=dataclasses.replace(transported,tracers={**transported.tracers,
            'co2_mmr_gas':1.-transported.tracers['ar_mmr_gas']})
    return transported,diagnostics
