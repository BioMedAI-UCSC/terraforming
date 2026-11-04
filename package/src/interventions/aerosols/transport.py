"""Positive conservative aerosol operators in actual sigma-layer gas masses.

CO2 Sutherland parameters: mu0=1.370e-5 Pa s, T0=273 K, S=222 K
(COMSOL High Mach Number Flow, Sutherland-law material table). Continuum
rod resistance follows Tirado et al. 1984, DOI 10.1063/1.447827. Free-molecular
resistance uses Epstein 1924, DOI 10.1103/PhysRev.23.710 and random projected
area. The harmonic resistance blend and aggregate radius scaling are explicit
engineering/sensitivity closures, not measured Mars particle drag.
"""
from __future__ import annotations

import math
import numpy as np

from src.framework.gcm._dinosaur import jax,jnp

K_B=1.380649e-23
R_CO2=188.92


def gas_layer_mass(ps_pa,sigma_boundaries,gravity):
    return jnp.diff(jnp.asarray(sigma_boundaries))[:,None,None]*ps_pa[None]/gravity


def co2_viscosity(temperature_k):
    return 1.370e-5*(temperature_k/273.)**1.5*(273.+222.)/(temperature_k+222.)


def particle_drag(particle,temperature_k,pressure_pa,aggregate_multiplicity=1.):
    mu=co2_viscosity(temperature_k)
    radius=particle.volume_radius_m
    if particle.shape == "rod":
        aspect=particle.aspect_ratio
        parallel=2*math.pi*particle.length_m/(math.log(aspect)-.207+.980/aspect-.133/aspect**2)
        perpendicular=4*math.pi*particle.length_m/(math.log(aspect)+.839+.185/aspect+.233/aspect**2)
        continuum=mu/(1/(3*parallel)+2/(3*perpendicular))
    else:
        # Disks require a measured mobility model for predictive drag; this
        # volume-equivalent sphere is explicitly a bounded screening closure.
        continuum=6*math.pi*mu*radius
    scale=jnp.asarray(aggregate_multiplicity)**(1/3)
    continuum=continuum*scale
    density=pressure_pa/(R_CO2*temperature_k)
    mean_speed=jnp.sqrt(8*R_CO2*temperature_k/math.pi)
    free=(4/3)*density*mean_speed*particle.mean_projected_area_m2*particle.accommodation*scale**2
    return 1/(1/continuum+1/free)


def terminal_velocity(particle,temperature_k,pressure_pa,gravity,aggregate_multiplicity=1.):
    return particle.mass_kg*aggregate_multiplicity*gravity/particle_drag(
        particle,temperature_k,pressure_pa,aggregate_multiplicity)


def implicit_settle(burden_kg_m2,number_m2,layer_gas_kg_m2,temperature_k,pressure_pa,dt_s,particle,gravity):
    """Implicit donor-cell settling; no artificial terminal-speed/CFL cap.

    Returns suspended mass/number and bottom deposition. The lower layer uses
    the *updated* flux from above, allowing large timesteps without negative
    burdens or deleting material. Accuracy still requires timestep refinement.
    """
    multiplicity=jnp.where(number_m2>0,burden_kg_m2/(jnp.maximum(number_m2,1e-300)*particle.mass_kg),1.)
    multiplicity=jnp.maximum(multiplicity,1.)
    density=pressure_pa/(R_CO2*temperature_k)
    thickness=layer_gas_kg_m2/density
    rates=terminal_velocity(particle,temperature_k,pressure_pa,gravity,multiplicity)/thickness
    def solve(field):
        def row(incoming,values):
            mass,rate=values
            new=(mass+incoming)/(1+dt_s*rate)
            out=dt_s*rate*new
            return out,new
        deposited,new=jax.lax.scan(row,jnp.zeros_like(field[0]),(field,rates))
        return new,deposited
    mass,deposit_mass=solve(burden_kg_m2)
    number,deposit_number=solve(number_m2)
    return mass,number,deposit_mass,deposit_number


def remove_fraction(mass,number,rate_s,dt_s):
    """Exact first-order sink applied to the remaining inventory once."""
    fraction=-jnp.expm1(-jnp.asarray(rate_s)*dt_s)
    removed_mass=mass*fraction
    removed_number=number*fraction
    return mass-removed_mass,number-removed_number,removed_mass,removed_number


def remap_carrier(mixing_ratio,old_gas_mass,new_gas_mass):
    """Changing carrier gas must not create/destroy particulate column mass."""
    return mixing_ratio*old_gas_mass/new_gas_mass


def positive_column_burden(burden):
    """Conservative column repair of spectral undershoots, with repair diagnostic.

    A column with negative total inventory is invalid (NaN), not silently
    clipped to a material-creating positive state.
    """
    total=jnp.sum(burden,axis=0,keepdims=True)
    positive=jnp.maximum(burden,0.)
    available=jnp.sum(positive,axis=0,keepdims=True)
    repaired=positive*jnp.where(available>0,total/jnp.maximum(available,1e-300),0.)
    repaired=jnp.where(total>=0,repaired,jnp.nan)
    return repaired,jnp.sum(jnp.abs(repaired-burden),axis=0)


def injection_pattern(source,coords):
    grid=coords.horizontal
    lat=np.degrees(np.asarray(grid.latitudes))[None,:]
    lon=np.mod(np.degrees(np.asarray(grid.longitudes))[:,None],360.)
    horizontal=(lat>=source.latitude_min_deg)&(lat<=source.latitude_max_deg)
    if source.longitude_min_deg < source.longitude_max_deg:
        horizontal=horizontal&(lon>=source.longitude_min_deg)&(lon<=source.longitude_max_deg)
    else:
        horizontal=horizontal&((lon>=source.longitude_min_deg)|(lon<=source.longitude_max_deg))
    horizontal=np.broadcast_to(horizontal,grid.nodal_shape).astype(float)
    weights=np.broadcast_to(np.asarray(grid.quadrature_weights),grid.nodal_shape).copy()
    weights/=weights.sum()
    area_norm=np.sum(horizontal*weights)
    centers=np.asarray(coords.vertical.centers)
    vertical=np.diff(np.asarray(coords.vertical.boundaries)).astype(float)
    vertical*=((centers>=source.sigma_min)&(centers<=source.sigma_max))
    if area_norm<=0 or vertical.sum()<=0:
        raise ValueError("source does not intersect the selected horizontal/vertical grid")
    return jnp.asarray(vertical[:,None,None]/vertical.sum()*horizontal[None]/area_norm)


def integrated_source(start_s,dt_s,sources,patterns,particle,area_m2):
    mass=jnp.zeros_like(patterns[0]) if patterns else None
    total=jnp.asarray(0.)
    for source,pattern in zip(sources,patterns,strict=True):
        overlap=jnp.maximum(0.,jnp.minimum(start_s+dt_s,source.stop_s)-jnp.maximum(start_s,source.start_s))
        injected=overlap*source.mass_rate(particle)
        mass=mass+injected/area_m2*pattern
        total=total+injected
    return mass,total
