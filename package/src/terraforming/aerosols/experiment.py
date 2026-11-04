"""Continuous particle/GCM stepping, with explicit inventory and validity gates.

Screening experiments are not forecasts of a validated terraformed climate.
All times and resource rates are SI; no implicit Earth/Mars-year conversion.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import math
import numpy as np
from src.framework.gcm._dinosaur import jax, jnp, scales
from src.framework.gcm.dynamics import stepper, reference_temperature
from src.framework.physics.gcm import forced_co2_primitive_equations, positivity_preserving_co2_step
from .state import (MASS, NUMBER, burdens, put_burdens, gas_mass, inventory,
                    correct_transport, quadrature_weights)
from .transport import injection_pattern, integrated_source, implicit_settle, remove_fraction, R_CO2
from .microphysics import brownian_kernel, constant_kernel_coagulation
from .radiation import radiation_component


@dataclass(frozen=True)
class ExperimentOptions:
    mode: str = 'passive'
    dt_s: float = 60.
    coagulation: bool = False
    charge_e: float = 0.
    collection_rate_s: float = 0.
    collection_source: str | None = None
    max_relative_transport_correction: float = 1e-4

    def __post_init__(self):
        if self.mode not in ('off','passive','active'):
            raise ValueError('mode must be off, passive or active')
        if not math.isfinite(self.dt_s) or not 0<self.dt_s<=300:
            raise ValueError('dt_s must be in (0,300] seconds')
        if not math.isfinite(self.charge_e):
            raise ValueError('charge must be finite')
        if not math.isfinite(self.collection_rate_s) or self.collection_rate_s<0:
            raise ValueError('collection rate must be finite and nonnegative')
        if self.collection_rate_s and not self.collection_source:
            raise ValueError('collection requires a named source or sensitivity assumption')
        if not 0<self.max_relative_transport_correction<=.01:
            raise ValueError('transport correction limit must be in (0,.01]')
        if self.mode=='active' and self.coagulation:
            raise ValueError('active aggregate optics are unavailable; use passive coagulation screening')


def coupled_step(coords,specs,body,forcing,co2_forcing,particle,sources,options,*,bands=None,orography=None):
    """Build one continuous split step; no atmosphere resets between sources.

    Injection/settling are first-order split operators. Refine dt for accuracy.
    The gas mass projection is retained from the base GCM; its carrier change
    is remapped separately so it cannot remove solid particles.
    """
    if options.mode=='active' and bands is None:
        raise ValueError('active mode requires verified shape/material optics')
    component=radiation_component(bands,forcing) if options.mode=='active' else None
    equation=forced_co2_primitive_equations(coords,body,forcing,co2_forcing,
        specs=specs,orography=orography,radiation_component=component,
        conserved_particle_tracers=(MASS,NUMBER))
    raw_step=stepper(equation,options.dt_s,specs)
    # Reuse the base projection, while retaining the unprojected particle burden.
    projected=lambda raw,old: positivity_preserving_co2_step(lambda _:raw,coords,specs)(old)
    patterns=tuple(injection_pattern(s,coords) for s in sources)
    dt=options.dt_s; grid=coords.horizontal
    weights=quadrature_weights(coords)*body.surface_area_m2
    temp_scale=float(specs.dimensionalize(1.,scales.units.kelvin).magnitude)
    def step(state):
        expected_m,expected_n=inventory(state,coords,specs,body)
        raw=raw_step(state.climate)
        mass,number=burdens(raw,coords,specs,body)
        climate=put_burdens(projected(raw,state.climate),mass,number,coords,specs,body)
        state=correct_transport(state._replace(climate=climate),expected_m,expected_n,
            coords,specs,body,max_relative_correction=options.max_relative_transport_correction)
        mass,number=burdens(state.climate,coords,specs,body)
        ledger=state.ledger
        if options.mode!='off' and sources:
            added,total=integrated_source(state.elapsed_seconds,dt,sources,patterns,particle,body.surface_area_m2)
            mass=mass+added; number=number+added/particle.mass_kg
            ledger=ledger._replace(injected_mass_kg=ledger.injected_mass_kg+total,
                injected_number=ledger.injected_number+total/particle.mass_kg)
        gas=gas_mass(state.climate,coords,specs,body)
        air=(grid.to_nodal(state.climate.dynamics.temperature_variation)+
            jnp.asarray(reference_temperature(coords,body))[:,None,None])*temp_scale
        ps=jnp.sum(gas,axis=0)*body.gravity_m_s2
        pressure=jnp.asarray(coords.vertical.centers)[:,None,None]*ps
        thickness=gas*R_CO2*air/pressure
        if options.coagulation:
            multiplicity=jnp.maximum(mass/jnp.maximum(number*particle.mass_kg,1e-300),1.)
            kernel=brownian_kernel(particle,air,pressure,multiplicity,options.charge_e)
            mass,number,_=constant_kernel_coagulation(mass,number,kernel,thickness,dt)
        mass,number,dm,dn=implicit_settle(mass,number,gas,air,pressure,dt,particle,body.gravity_m_s2)
        ledger=ledger._replace(deposited_mass_kg_m2=ledger.deposited_mass_kg_m2+dm,
            deposited_number_m2=ledger.deposited_number_m2+dn)
        mass,number,cm,cn=remove_fraction(mass,number,options.collection_rate_s,dt)
        ledger=ledger._replace(collected_mass_kg_m2=ledger.collected_mass_kg_m2+jnp.sum(cm,axis=0),
            collected_number_m2=ledger.collected_number_m2+jnp.sum(cn,axis=0))
        invalid=(jnp.any(~jnp.isfinite(air))|jnp.any((air<100.)|(air>350.))|
            jnp.any((pressure<1.)|(pressure>2000.))|
            jnp.any(mass/gas>particle.max_mass_loading)|jnp.any(~jnp.isfinite(mass))|
            jnp.any(~jnp.isfinite(number))|jnp.any(number<0))
        ledger=ledger._replace(invalid=ledger.invalid|invalid)
        return state._replace(climate=put_burdens(state.climate,mass,number,coords,specs,body),
            ledger=ledger,elapsed_seconds=state.elapsed_seconds+dt)
    return jax.jit(step)


@jax.jit
def _finite_climate(climate):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(climate)]))


def diagnostics(state,coords,specs,body,particle):
    mass,number=inventory(state,coords,specs,body)
    w=quadrature_weights(coords)*body.surface_area_m2
    deposited=jnp.sum(state.ledger.deposited_mass_kg_m2*w)
    collected=jnp.sum(state.ledger.collected_mass_kg_m2*w)
    expected=state.ledger.initial_mass_kg+state.ledger.injected_mass_kg
    surface=jnp.asarray(state.climate.surface_temperature)[0]
    return dict(elapsed_seconds=float(state.elapsed_seconds),suspended_mass_kg=float(mass),
        suspended_number=float(number),deposited_mass_kg=float(deposited),collected_mass_kg=float(collected),
        deposited_number=float(jnp.sum(state.ledger.deposited_number_m2*w)),
        collected_number=float(jnp.sum(state.ledger.collected_number_m2*w)),
        injected_mass_kg=float(state.ledger.injected_mass_kg),
        injected_solid_volume_m3=float(state.ledger.injected_mass_kg)/particle.density_kg_m3,
        mean_aggregate_multiplicity=None if float(number)==0 else float(mass/number)/particle.mass_kg,
        aggregate_drag_assumption='compact volume scaling; screening closure, not measured aggregate mobility',
        source_energy_momentum_assumption='release equilibrates with local air; industrial/release plume energy and momentum are not resolved',
        material_budget_error_kg=float(mass+deposited+collected-expected),
        maximum_raw_relative_mass_error=float(state.ledger.maximum_raw_relative_mass_error),
        numerical_mass_correction_kg=float(state.ledger.numerical_mass_correction_kg),
        surface_temperature_k=float(jnp.sum(surface*w)/body.surface_area_m2),
        production_energy_j=None if particle.production_energy_j_kg is None else
            float(state.ledger.injected_mass_kg)*particle.production_energy_j_kg,
        status=('numerical_failure' if not bool(_finite_climate(state.climate)) else
            'invalid_regime' if bool(state.ledger.invalid) else 'running'),
        baseline_readiness='unverified: requires matched climatology and resolution checks')
