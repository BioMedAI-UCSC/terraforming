"""JAX adaptation of Ames v3.2.1 update_mars_atmos.F90::pass2.

NASA Ames-derived algorithms: Copyright 2023 United States Government as
represented by the Administrator of NASA. NASA Open Source Agreement 1.3;
see iclr-results/06-co2-physics/ames-reference/NOSA.pdf in this repository.

The sweep order, stability test and pressure weights follow the reference.
Only array layout and control flow change for batched JAX execution. Inputs
are the predicted physical state, after the other physics tendencies act.
"""
from __future__ import annotations

from src.framework.physics.tracer_transport import tracer_to_nodal,tracer_from_nodal
import dataclasses
import numpy as np

from src.framework.gcm._dinosaur import jax, jnp, spherical_harmonic, scales
from src.framework.gcm.dynamics import reference_temperature


def pass2(temperature, u, v, tracers, pressure_half, pressure_full, kappa):
    """One bottom-to-top Ames sweep; levels first, tracer species last.

    Shapes are ``(level, *columns)`` and ``(level, *columns, species)``.
    All inputs use SI units. The branch at neutral stability is piecewise
    differentiable, just as the original conditional adjustment.
    """
    temperature, u, v, tracers, pressure_half, pressure_full = map(
        jnp.asarray, (temperature, u, v, tracers, pressure_half, pressure_full))
    if temperature.ndim < 1 or temperature.shape[0] < 2:
        raise ValueError("convection requires at least two vertical layers")
    if u.shape != temperature.shape or v.shape != temperature.shape:
        raise ValueError("temperature and winds must have the same shape")
    if tracers.shape[:-1] != temperature.shape:
        raise ValueError("tracers must have temperature shape plus a species axis")
    if pressure_full.shape != temperature.shape or pressure_half.shape != (
            temperature.shape[0] + 1, *temperature.shape[1:]):
        raise ValueError("midpoint and interface pressures must match the column grid")
    delp = jnp.diff(pressure_half, axis=0)

    dtdp = .5*kappa*jnp.diff(pressure_full,axis=0)/pressure_half[1:-1]

    def sweep(lower, inputs):
        tu,uu,vu,ru,d,a,b = inputs
        tl,ul,vl,rl = lower
        alf = (1.+d)/(1.-d)
        rfrac = 1./(a+b*alf)
        rfrac2 = 1./(a+b)
        unstable = (d-1.)*tl+(d+1.)*tu < 0.
        upper = (a*tu+b*tl)*rfrac
        tupper = jnp.where(unstable,upper,tu)
        tlower = jnp.where(unstable,upper*alf,tl)
        umix = (a*uu+b*ul)*rfrac2
        vmix = (a*vu+b*vl)*rfrac2
        rmix = (a[...,None]*ru+b[...,None]*rl)*rfrac2[...,None]
        next_upper = (tupper,jnp.where(unstable,umix,uu),
            jnp.where(unstable,vmix,vu),jnp.where(unstable[...,None],rmix,ru))
        fixed_lower = (tlower,jnp.where(unstable,umix,ul),
            jnp.where(unstable,vmix,vl),jnp.where(unstable[...,None],rmix,rl))
        return next_upper,fixed_lower

    # Only the adjacent lower layer changes before the next interface is read.
    # Carry it through the scan instead of copying full columns on every pair.
    initial = (temperature[-1],u[-1],v[-1],tracers[-1])
    inputs = tuple(x[::-1] for x in (temperature[:-1],u[:-1],v[:-1],tracers[:-1],
        dtdp,delp[:-1],delp[1:]))
    top,lower = jax.lax.scan(sweep,initial,inputs)
    return tuple(jnp.concatenate([first[None],tail[::-1]],axis=0)
                 for first,tail in zip(top,lower,strict=True))


def adjust_state(state, coords, specs, body):
    """Apply pass2 to a predicted Dinosaur state, preserving unchanged modes."""
    grid = coords.horizontal
    dyn = state.dynamics
    temperature = grid.to_nodal(dyn.temperature_variation)
    reference = jnp.asarray(reference_temperature(coords, body))[:, None, None]
    temperature = temperature + reference
    velocity_scale = float(specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
    pressure_scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    u, v = spherical_harmonic.vor_div_to_uv_nodal(grid, dyn.vorticity, dyn.divergence)
    u, v = u * velocity_scale, v * velocity_scale
    ps = jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0] * pressure_scale
    half = jnp.asarray(np.asarray(coords.vertical.boundaries))[:, None, None] * ps
    full = jnp.asarray(np.asarray(coords.vertical.centers))[:, None, None] * ps
    names = tuple(dyn.tracers)
    tracers = (jnp.stack([tracer_to_nodal(grid,dyn.tracers[name]) for name in names], axis=-1)
               if names else jnp.empty(temperature.shape + (0,), dtype=temperature.dtype))
    t, unew, vnew, r = pass2(temperature, u, v, tracers, half, full, body.kappa)
    vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(
        grid, (unew-u)/velocity_scale, (vnew-v)/velocity_scale)
    return state._replace(dynamics=dataclasses.replace(dyn,
        temperature_variation=dyn.temperature_variation + grid.to_modal(t-temperature),
        vorticity=dyn.vorticity + vor, divergence=dyn.divergence + div,
        tracers={name:tracer_from_nodal(grid,r[...,i],dyn.tracers[name])
                 for i, name in enumerate(names)}))


def build_step(coords, specs, body, forcing, co2_forcing, dt_seconds, *,
               orography=None, hyperdiffusion_tau_seconds=None):
    """Build a Dinosaur/CO2 step followed by the exact Ames convection sweep.

    Disable the older heat-only convection to avoid applying both schemes.
    Other physics remains the framework implementation; this factory does not
    establish Ames equivalence for radiation, PBL or the CO2 cloud scheme.
    """
    from src.framework.gcm.dynamics import stepper
    from src.framework.physics.gcm import (
        forced_co2_primitive_equations, positivity_preserving_co2_step)
    forcing = dataclasses.replace(forcing,convective_adjustment_enabled=False)
    equation = forced_co2_primitive_equations(coords,body,forcing,co2_forcing,
        specs=specs,orography=orography)
    raw = stepper(equation,dt_seconds,specs,coords=coords,
        hyperdiffusion_tau_seconds=hyperdiffusion_tau_seconds)
    physical = positivity_preserving_co2_step(raw,coords,specs,body=body,
        co2_forcing=co2_forcing,dt_seconds=dt_seconds)

    def advance(state):
        return adjust_state(physical(state),coords,specs,body)

    return advance
