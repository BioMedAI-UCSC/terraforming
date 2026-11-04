"""Pulse-release lifetime sensitivities in an explicitly well-mixed column.

This isolated model holds gas pressure, temperature and mixing depth fixed.
It separates self-coagulation, collector attachment, gravitational deposition,
and unresolved suspended products. It is not a validated Mars retention model.
All times are SI; the roadmap's unqualified year requires an explicit choice.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp
from .specs import ParticleSpec
from .transport import K_B,R_CO2,particle_drag
from .microphysics import charge_factor
from .sectional import coagulator,collision_kernels


@dataclass(frozen=True)
class CollectorScenario:
    radius_m: float
    density_kg_m3: float
    number_m3: float
    provenance: str
    efficiency: float = 1.
    charge_e: float = 0.

    def __post_init__(self):
        if not self.provenance.strip(): raise ValueError('collector size/number scenario requires provenance')
        if not all(math.isfinite(x) for x in (self.radius_m,self.density_kg_m3,self.number_m3,self.efficiency,self.charge_e)):
            raise ValueError('collector inputs must be finite')
        if self.radius_m<=0 or self.density_kg_m3<=0 or self.number_m3<0 or not 0<=self.efficiency<=1:
            raise ValueError('invalid collector size, density, number or efficiency')
        try: mass=4*math.pi/3*self.radius_m**3*self.density_kg_m3
        except OverflowError as exc: raise ValueError('collector mass is outside numeric range') from exc
        if not math.isfinite(mass) or mass<=0: raise ValueError('collector mass is outside numeric range')


class Population(NamedTuple):
    number_m3: object
    unresolved_mass_kg_m3: object
    unresolved_number_m3: object
    deposited_mass_kg_m3: object
    collected_mass_kg_m3: object
    prescribed_removed_mass_kg_m3: object
    integrated_fractions_s: object
    elapsed_s: object
    last_valid_s: object
    numerical: object
    invalid: object


def _collector_rate(particle,mass,temperature,pressure,gravity,charge,dimension,collector):
    ratio=mass/particle.mass_kg
    radius=particle.volume_radius_m*ratio**(1/dimension)
    drag=particle_drag(particle,temperature,pressure,ratio**(3/dimension))
    dust=ParticleSpec('prescribed_collector',collector.density_kg_m3,
        2*collector.radius_m,2*collector.radius_m,2*collector.radius_m,
        shape='sphere',source=collector.provenance)
    dust_drag=particle_drag(dust,temperature,pressure)
    separation=radius+collector.radius_m
    diffusive=4*math.pi*separation*K_B*temperature*(1/drag+1/dust_drag)
    ballistic=math.pi*separation**2*jnp.sqrt(8*K_B*temperature/math.pi*(1/mass+1/dust.mass_kg))
    thermal=1/(1/diffusive+1/ballistic)*charge_factor(charge,collector.charge_e,radius,collector.radius_m,temperature)
    # Neutral capture only: no unvalidated charged/photophoretic force law.
    gravity_kernel=math.pi*separation**2*jnp.abs(mass*gravity/drag-dust.mass_kg*gravity/dust_drag)
    return collector.number_m3*collector.efficiency*(thermal+gravity_kernel)


def lifetime_screen(particle,*,horizon_s,dt_s,column_mass_kg_m2,mixing_depth_m,
                    temperature_k,pressure_pa,gravity_m_s2,year_convention,
                    max_monomers=32,charge_e=0.,sticking_efficiency=1.,
                    fractal_dimension=3.,settling_efficiency=0.,collector=None,
                    removal_mode='resolved',prescribed_lifetime_s=None,
                    prescribed_source=None,max_unresolved_mass_fraction=1e-6):
    """Return finite-horizon residence integrals and uncertainty assumptions.

    Mass/number/monomer results are lower bounds on pulse residence times when
    the final free inventory is nonzero. Collector-bound material is counted in
    its own reservoir; its subsequent transport and optics are not inferred.
    Aggregate radiative efficacy is unknown and is never replaced by mass.
    """
    values=(horizon_s,dt_s,column_mass_kg_m2,mixing_depth_m,temperature_k,pressure_pa,gravity_m_s2)
    if not all(math.isfinite(x) and x>0 for x in values): raise ValueError('positive finite physical inputs required')
    if year_convention=='earth_julian': year_s=365.25*86400.
    elif year_convention=='mars_orbital':
        from src.celestials.planets.mars.planet import MARS_ORBITAL_PERIOD
        year_s=float(MARS_ORBITAL_PERIOD)
    else: raise ValueError('choose earth_julian or mars_orbital explicitly; the roadmap year is ambiguous')
    if not isinstance(max_monomers,int) or not 4<=max_monomers<=128: raise ValueError('size grid must have 4..128 monomer-mass pivots')
    if not 0<max_unresolved_mass_fraction<=.01: raise ValueError('unresolved mass tolerance must be in (0,.01]')
    if removal_mode not in ('resolved','prescribed'): raise ValueError('unknown removal mode')
    if removal_mode=='prescribed':
        if prescribed_lifetime_s is None or not math.isfinite(prescribed_lifetime_s) or prescribed_lifetime_s<=0 or not prescribed_source:
            raise ValueError('prescribed removal requires a positive SI lifetime and its source/assumption')
        if sticking_efficiency!=0 or collector is not None or settling_efficiency!=0:
            raise ValueError('prescribed lifetime must be separate from resolved coagulation/collection/removal')
    elif prescribed_lifetime_s is not None: raise ValueError('a prescribed lifetime cannot be added to resolved removal')
    if collector is not None and collector.number_m3>0 and (charge_e!=0 or collector.charge_e!=0):
        raise ValueError('charged collector capture requires a validated nonthermal capture law')
    gas_density=pressure_pa/(R_CO2*temperature_k)
    initial_mass=column_mass_kg_m2/mixing_depth_m
    if not math.isfinite(initial_mass) or initial_mass<=np.finfo(float).tiny:
        raise ValueError('pulse density is outside numeric range')
    dust_density=0. if collector is None else collector.number_m3*(4*math.pi/3*collector.radius_m**3*collector.density_kg_m3)
    if (initial_mass+dust_density)/gas_density>particle.max_mass_loading:
        raise ValueError('pulse exceeds dilute particle loading')
    pivots=particle.mass_kg*np.arange(1.,max_monomers+1.)
    brownian,settling=collision_kernels(particle,pivots,temperature_k,pressure_pa,gravity_m_s2,
        charges_e=charge_e,sticking_efficiency=sticking_efficiency,fractal_dimension=fractal_dimension,
        settling_efficiency=settling_efficiency)
    kernel=brownian+settling;mass=jnp.asarray(pivots);coagulate=coagulator(pivots)
    initial_number=initial_mass/particle.mass_kg
    initial=jnp.zeros(max_monomers).at[0].set(initial_number)
    drag=particle_drag(particle,temperature_k,pressure_pa,(mass/particle.mass_kg)**(3/fractal_dimension))
    deposit_rate=mass*gravity_m_s2/drag/mixing_depth_m
    collection_rate=jnp.zeros(max_monomers) if collector is None else _collector_rate(
        particle,mass,temperature_k,pressure_pa,gravity_m_s2,charge_e,fractal_dimension,collector)
    if removal_mode=='prescribed': deposit_rate=jnp.full(max_monomers,1/prescribed_lifetime_s)
    count=math.ceil(horizon_s/dt_s)
    if count>2_000_000: raise ValueError('screening step budget exceeded; choose an explicit coarser sensitivity grid')
    zero=jnp.asarray(0.,dtype=initial.dtype)
    state=Population(number_m3=initial,unresolved_mass_kg_m3=zero,unresolved_number_m3=zero,
        deposited_mass_kg_m3=zero,collected_mass_kg_m3=zero,prescribed_removed_mass_kg_m3=zero,
        integrated_fractions_s=jnp.zeros(3),elapsed_s=zero,last_valid_s=zero,
        numerical=jnp.asarray(False),invalid=jnp.asarray(False))

    def fractions(s):
        return jnp.array([(jnp.dot(s.number_m3,mass)+s.unresolved_mass_kg_m3)/initial_mass,
            (jnp.sum(s.number_m3)+s.unresolved_number_m3)/initial_number,s.number_m3[0]/initial_number])

    def advance(s,index):
        end=jnp.minimum(horizon_s,(index+1)*dt_s)
        dt=jnp.maximum(0.,end-s.elapsed_s)
        if removal_mode=='resolved':
            product=coagulate(s.number_m3,kernel,dt)
            number=product.number_m3
            overflow_mass=s.unresolved_mass_kg_m3+product.unresolved_mass_kg_m3
            overflow_number=s.unresolved_number_m3+product.unresolved_number_m3
            valid=product.valid
        else:
            number=s.number_m3;overflow_mass=s.unresolved_mass_kg_m3;overflow_number=s.unresolved_number_m3
            valid=jnp.asarray(True)
        deposited=number*(-jnp.expm1(-deposit_rate*dt))
        number=number-deposited
        collected=number*(-jnp.expm1(-collection_rate*dt))
        number=number-collected
        candidate=s._replace(number_m3=number,unresolved_mass_kg_m3=overflow_mass,
            unresolved_number_m3=overflow_number,
            deposited_mass_kg_m3=s.deposited_mass_kg_m3+(jnp.dot(deposited,mass) if removal_mode=='resolved' else 0.),
            prescribed_removed_mass_kg_m3=s.prescribed_removed_mass_kg_m3+(jnp.dot(deposited,mass) if removal_mode=='prescribed' else 0.),
            collected_mass_kg_m3=s.collected_mass_kg_m3+jnp.dot(collected,mass),elapsed_s=end)
        finite=jnp.all(jnp.isfinite(number))&jnp.isfinite(overflow_mass)&jnp.isfinite(overflow_number)
        invalid=(~valid)|(overflow_mass>max_unresolved_mass_fraction*initial_mass)
        candidate=candidate._replace(integrated_fractions_s=s.integrated_fractions_s+dt*(fractions(s)+fractions(candidate))/2,
            numerical=~finite,invalid=invalid,last_valid_s=jnp.where(finite&~invalid,candidate.elapsed_s,s.last_valid_s))
        return jax.lax.cond(finite,lambda _:candidate,
            lambda _:s._replace(numerical=jnp.asarray(True)),operand=None)

    def row(s,index):
        return jax.lax.cond(s.numerical|s.invalid,lambda current:current,
            lambda current:advance(current,index),s),None
    final,_=jax.lax.scan(row,state,jnp.arange(count))
    suspended=float(jnp.dot(final.number_m3,mass)+final.unresolved_mass_kg_m3)*mixing_depth_m
    deposited=float(final.deposited_mass_kg_m3)*mixing_depth_m
    collected=float(final.collected_mass_kg_m3)*mixing_depth_m
    prescribed=float(final.prescribed_removed_mass_kg_m3)*mixing_depth_m
    status='numerical_failure' if bool(final.numerical) else 'invalid_regime' if bool(final.invalid) else 'completed'
    integrals=np.asarray(final.integrated_fractions_s)
    return dict(status=status,simulated_s=float(final.elapsed_s),last_valid_s=float(final.last_valid_s),
        suspended_mass_kg_m2=suspended,deposited_mass_kg_m2=deposited,collected_mass_kg_m2=collected,
        prescribed_removed_mass_kg_m2=prescribed,
        material_budget_error_kg_m2=suspended+deposited+collected+prescribed-column_mass_kg_m2,
        unresolved_suspended_mass_kg_m2=float(final.unresolved_mass_kg_m3)*mixing_depth_m,
        final_number_m3=np.asarray(final.number_m3).tolist(),mass_pivots_kg=pivots.tolist(),
        finite_horizon_residence_lower_bounds_s=None if status!='completed' else dict(zip(('mass','number','monomers'),integrals.tolist())),
        radiatively_effective_residence_time_s=None,
        roadmap_criterion_s=year_s/3,roadmap_year_convention=year_convention,
        roadmap_effective_lifetime_comparison='not evaluable without aggregate optical efficacy and climate validation',
        right_censored=bool(float(fractions(final)[0])>1e-6),
        assumptions=dict(model='isolated fixed-gas well-mixed column; prescribed pulse, no climate feedback',
            dt_s=dt_s,horizon_s=horizon_s,max_monomers=max_monomers,
            monomer_deposition_or_prescribed_rate_s=float(deposit_rate[0]),monomer_collection_rate_s=float(collection_rate[0]),
            monomer_source=particle.source,temperature_k=temperature_k,pressure_pa=pressure_pa,mixing_depth_m=mixing_depth_m,
            charge_e=charge_e,sticking_efficiency=sticking_efficiency,fractal_dimension=fractal_dimension,
            aggregate_mobility='effective radius ~ monomer count**(1/D), continuum/Epstein screening interpolation',
            aggregate_optics='not supplied; no aggregate warming prediction',
            collector=None if collector is None else collector.__dict__,
            collected_material='collector-bound unresolved reservoir; not automatically surface deposition',
            removal_mode=removal_mode,prescribed_lifetime_s=prescribed_lifetime_s,prescribed_source=prescribed_source))
