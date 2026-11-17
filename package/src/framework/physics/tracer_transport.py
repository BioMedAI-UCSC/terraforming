"""Material-conserving positive projection for Dinosaur's spectral tracers.

This is a Dinosaur numerical adaptation. It is not the reference FV3 transport
scheme. Inventory errors and redistributed material remain separate diagnostics.
"""
import dataclasses
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


def project_tracers(previous,current,coords,*,conserve_previous=True):
    """Preserve each global species inventory and remove spectral undershoots.

    After dynamics, use the previous inventory. After physics, keep the current
    inventory so physical sources and sinks remain intact. A convex blend with
    the mass-weighted global mean stays in the same spectral space. The largest
    possible blend factor keeps all Gaussian-grid values nonnegative. Negative
    total inventory is invalid and returns NaNs instead of creating material.
    Diagnostic integrals use nondimensional pressure and Gaussian weights.
    """
    if set(previous.tracers)!=set(current.tracers):
        raise ValueError('transport projection requires the same tracer names before and after the step')
    grid=coords.horizontal
    ds=jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]
    carrier=jnp.exp(grid.to_nodal(current.log_surface_pressure))*ds*grid.quadrature_weights
    old_carrier=jnp.exp(grid.to_nodal(previous.log_surface_pressure))*ds*grid.quadrature_weights
    uniform=grid.to_modal(jnp.ones(grid.nodal_shape))
    tracers={};diagnostics={}
    for name,value in current.tracers.items():
        nodal=grid.to_nodal(value)
        raw=jnp.sum(nodal*carrier)
        target=jnp.sum(grid.to_nodal(previous.tracers[name])*old_carrier) if conserve_previous else raw
        def repair(_):
            mean=raw/jnp.sum(carrier)
            minimum=jnp.min(nodal)
            margin=16.*jnp.finfo(nodal.dtype).eps*jnp.max(jnp.abs(nodal))
            theta=jnp.clip((mean-margin)/jnp.where(mean>minimum,mean-minimum,1.),0.,1.)
            return value*theta+uniform*(mean*(1.-theta))
        positive=jax.lax.cond(jnp.any(nodal<0.),repair,lambda _:value,None)
        actual=jnp.sum(grid.to_nodal(positive)*carrier)
        scale=target/jnp.where(actual>0.,actual,1.)
        result=jnp.where(target==0.,jnp.zeros_like(value),positive*scale)
        valid=(target>=0.)&((actual>0.)|(target==0.))&(raw>=0.)
        tracers[name]=jnp.where(valid,result,jnp.nan)
        change=grid.to_nodal(tracers[name])-nodal
        diagnostics[name]=dict(raw_inventory=raw,target_inventory=target,
            corrected_inventory=jnp.sum(grid.to_nodal(tracers[name])*carrier),
            raw_inventory_change=raw-target,redistributed_inventory=jnp.sum(jnp.abs(change)*carrier),
            minimum_before=jnp.min(nodal),minimum_after=jnp.min(grid.to_nodal(tracers[name])),valid=valid)
    return dataclasses.replace(current,tracers=tracers),diagnostics
