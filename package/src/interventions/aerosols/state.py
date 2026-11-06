"""Particle reservoirs and diagnosed numerical corrections, separate from gas."""
from __future__ import annotations

from typing import NamedTuple
import dataclasses
import numpy as np
from src.framework.gcm._dinosaur import jnp,scales
from .transport import gas_layer_mass,positive_column_burden

MASS='engineered_aerosol_mass'
NUMBER='engineered_aerosol_number'


class AerosolLedger(NamedTuple):
    deposited_mass_kg_m2: object
    deposited_number_m2: object
    collected_mass_kg_m2: object
    collected_number_m2: object
    initial_mass_kg: object
    injected_mass_kg: object
    injected_number: object
    re_lofted_mass_kg: object
    numerical_mass_correction_kg: object
    numerical_number_correction: object
    positivity_redistribution_kg: object
    maximum_raw_relative_mass_error: object
    invalid: object
    numerical_dry_energy_correction_j: object
    numerical_frost_latent_correction_j: object
    last_unprojected_dry_energy_j: object
    last_projected_dry_energy_j: object


class AerosolState(NamedTuple):
    climate: object
    ledger: AerosolLedger
    elapsed_seconds: object


def quadrature_weights(coords):
    w=np.broadcast_to(np.asarray(coords.horizontal.quadrature_weights),coords.horizontal.nodal_shape).copy()
    return jnp.asarray(w/w.sum())


def gas_mass(state,coords,specs,body):
    pressure=jnp.exp(coords.horizontal.to_nodal(state.dynamics.log_surface_pressure))[0]
    pressure*=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    return gas_layer_mass(pressure,coords.vertical.boundaries,body.gravity_m_s2)


def burdens(state,coords,specs,body):
    grid=coords.horizontal
    gas=gas_mass(state,coords,specs,body)
    return grid.to_nodal(state.dynamics.tracers[MASS])*gas,grid.to_nodal(state.dynamics.tracers[NUMBER])*gas


def put_burdens(state,mass,number,coords,specs,body,*,positive=False):
    """Store tracers, optionally limiting their represented nodal values conservatively."""
    gas=gas_mass(state,coords,specs,body)
    grid=coords.horizontal
    weights=quadrature_weights(coords)*gas
    def modal(burden):
        coefficients=grid.to_modal(burden/gas)
        if not positive:
            return coefficients
        nodal=grid.to_nodal(coefficients)
        total=jnp.sum(burden*quadrature_weights(coords),axis=(1,2),keepdims=True)
        represented=jnp.sum(nodal*weights,axis=(1,2),keepdims=True)
        mean=represented/jnp.sum(weights,axis=(1,2),keepdims=True)
        minimum=jnp.min(nodal,axis=(1,2),keepdims=True)
        # Contract spectral deviations toward the gas-weighted mean, not a clipped field.
        # shortcut: contraction diffuses horizontal structure; use validated positive transport for predictions.
        theta=jnp.minimum(1.,(1.-1e-12)*mean/jnp.maximum(mean-minimum,1e-300))
        limited=theta*coefficients+grid.to_modal(jnp.broadcast_to((1.-theta)*mean,nodal.shape))
        scale=jnp.where(total==0,0.,total/jnp.maximum(represented,1e-300))
        valid=(total>=0)&((represented>0)|(total==0))
        return jnp.where(valid,limited*scale,jnp.nan)
    tracers=dict(state.dynamics.tracers)
    tracers[MASS]=modal(mass)
    tracers[NUMBER]=modal(number)
    return state._replace(dynamics=dataclasses.replace(state.dynamics,tracers=tracers))


def initialize(climate,coords,specs,body,particle,initial_column_mass_kg_m2=0.):
    if any(k in climate.dynamics.tracers for k in (MASS,NUMBER)):
        raise ValueError('particle tracers already exist: use experiment resume, not initialization')
    if not np.isfinite(initial_column_mass_kg_m2) or initial_column_mass_kg_m2<0:
        raise ValueError('initial burden must be finite and nonnegative')
    gas=gas_mass(climate,coords,specs,body)
    mass=gas/jnp.sum(gas,axis=0)*initial_column_mass_kg_m2
    climate=put_burdens(climate,mass,mass/particle.mass_kg,coords,specs,body,positive=True)
    surface=jnp.zeros(coords.horizontal.nodal_shape)
    zero=jnp.asarray(0.)
    initial=initial_column_mass_kg_m2*body.surface_area_m2
    ledger=AerosolLedger(
        deposited_mass_kg_m2=surface,deposited_number_m2=surface,
        collected_mass_kg_m2=surface,collected_number_m2=surface,
        initial_mass_kg=jnp.asarray(initial),injected_mass_kg=zero,injected_number=zero,
        re_lofted_mass_kg=zero,numerical_mass_correction_kg=zero,
        numerical_number_correction=zero,positivity_redistribution_kg=zero,
        maximum_raw_relative_mass_error=zero,invalid=jnp.asarray(False),
        numerical_dry_energy_correction_j=zero,numerical_frost_latent_correction_j=zero,
        last_unprojected_dry_energy_j=zero,last_projected_dry_energy_j=zero)
    return AerosolState(climate,ledger,zero)


def inventory(state,coords,specs,body):
    mass,number=burdens(state.climate,coords,specs,body)
    w=quadrature_weights(coords)*body.surface_area_m2
    return jnp.sum(mass*w),jnp.sum(number*w)


def correct_transport(state,expected_mass_kg,expected_number,coords,specs,body,*,max_relative_correction=1e-4):
    """Expose spectral transport error; do not relabel a fixer as verification.

    Redistribute negative undershoots within their columns before a bounded
    global roundoff/material correction. Large errors invalidate the run. The
    signed correction and maximum *raw* error remain in the ledger.
    """
    mass,number=burdens(state.climate,coords,specs,body)
    mass,redistributed=positive_column_burden(mass)
    number,_=positive_column_burden(number)
    w=quadrature_weights(coords)*body.surface_area_m2
    raw_mass=jnp.sum(mass*w)
    raw_number=jnp.sum(number*w)
    mass_error=expected_mass_kg-raw_mass
    number_error=expected_number-raw_number
    relative=jnp.abs(mass_error)/jnp.maximum(jnp.abs(expected_mass_kg),1e-30)
    number_relative=jnp.abs(number_error)/jnp.maximum(jnp.abs(expected_number),1e-30)
    mass*=jnp.where(raw_mass>0,expected_mass_kg/jnp.maximum(raw_mass,1e-300),1.)
    number*=jnp.where(raw_number>0,expected_number/jnp.maximum(raw_number,1e-300),1.)
    invalid=(~jnp.isfinite(relative))|(~jnp.isfinite(number_relative))|(relative>max_relative_correction)|(number_relative>max_relative_correction)
    climate=put_burdens(state.climate,mass,number,coords,specs,body,positive=True)
    stored_mass,_=burdens(climate,coords,specs,body)
    ledger=state.ledger._replace(
        numerical_mass_correction_kg=state.ledger.numerical_mass_correction_kg+mass_error,
        numerical_number_correction=state.ledger.numerical_number_correction+number_error,
        positivity_redistribution_kg=state.ledger.positivity_redistribution_kg+jnp.sum(redistributed*w)+jnp.sum(jnp.abs(stored_mass-mass)*w),
        maximum_raw_relative_mass_error=jnp.maximum(relative,state.ledger.maximum_raw_relative_mass_error),
        invalid=state.ledger.invalid|invalid)
    return state._replace(climate=climate,ledger=ledger)
