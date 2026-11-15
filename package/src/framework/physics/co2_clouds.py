"""Finite-volume CO2 ice transport and resolved exchange budgets, in SI units.

Ice shares the horizontal gas velocity and temperature (the drag/thermal
equilibrium limit). Its mass is a transported kg-ice/kg-gas tracer. Settling is
implicit donor-cell transport with an open lower boundary and closed top.
Kuroda et al. 2013, doi:10.1002/grl.50326, gives the Stokes/slip settling law
and 1600 kg/m3 ice density. Particle radius is a model input, not an inferred
microphysical size distribution. No ice optical properties are implied.
"""
from __future__ import annotations

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp

CO2_CLOUD_TRACER = "co2_cloud_ice"


def geopotential_matrix(sigma, gas_constant):
    """Hydrostatic center heights as a linear function of layer temperatures."""
    sigma = np.asarray(sigma, dtype=float)
    matrix = np.zeros((len(sigma), len(sigma)))
    matrix[-1, -1] = gas_constant * np.log(1. / sigma[-1])
    for k in range(len(sigma) - 2, -1, -1):
        matrix[k] = matrix[k + 1]
        increment = .5 * gas_constant * np.log(sigma[k + 1] / sigma[k])
        matrix[k, k:k + 2] += increment
    return jnp.asarray(matrix)


def _thermal_solve(rhs, capacity, ice, geometry):
    """Solve the triangular heat-capacity/ice-height system, bottom to top."""
    temperature = jnp.zeros_like(rhs)
    for k in range(rhs.shape[0] - 1, -1, -1):
        lower = jnp.sum(geometry[k, :, None, None] * temperature, axis=0)
        temperature = temperature.at[k].set(
            (rhs[k] - ice[k] * lower) / (capacity[k] + ice[k] * geometry[k, k]))
    return temperature


def layer_properties(temperature, ps_pa, ice_ratio, sigma, dsigma, body, cf):
    gas = ps_pa[None] * dsigma[:, None, None] / body.gravity_m_s2
    ice = gas * ice_ratio
    geometry = geopotential_matrix(sigma, body.gas_constant_j_kg_k)
    phi = jnp.einsum("ij,jxy->ixy", geometry, temperature)
    gas_enthalpy = body.cp_j_kg_k * temperature
    ice_enthalpy = cf.cloud_ice_heat_capacity_j_kg_k * temperature - cf.latent_heat_j_kg + phi
    capacity = body.cp_j_kg_k * gas + cf.cloud_ice_heat_capacity_j_kg_k * ice
    return gas, ice, geometry, gas_enthalpy, ice_enthalpy, capacity


def phase_exchange(temperature, ps_pa, ice_ratio, wind, frost_k, sigma, dsigma, body, cf):
    """Gas/ice exchange with sigma-grid redistribution and heat/momentum fluxes.

    Gas mass moves between sigma layers when condensation changes surface
    pressure. Donor-cell fluxes carry the same enthalpy and momentum as that
    mass. Warm ice can sublimate; only available condensate can return to gas.
    Returned rates conserve column mass, enthalpy+ice potential+kinetic energy,
    and horizontal momentum. Dry-core truncation errors are outside this operator.
    """
    gas, ice, geometry, hg, hi, capacity = layer_properties(
        temperature, ps_pa, ice_ratio, sigma, dsigma, body, cf)
    latent = cf.latent_heat_j_kg + (body.cp_j_kg_k - cf.cloud_ice_heat_capacity_j_kg_k) * temperature
    desired = capacity * (frost_k - temperature) / latent / cf.atmospheric_condensation_timescale_s
    phase = jnp.clip(desired, -jnp.maximum(ice, 0.) / cf.exchange_timestep_s,
                     gas / cf.exchange_timestep_s)
    total = jnp.sum(phase, axis=0)
    dgas = -dsigma[:, None, None] * total
    # Positive interface flux points down; boundary fluxes are exactly zero.
    flux = jnp.cumsum(-phase - dgas, axis=0)[:-1]
    def divergence(values):
        return jnp.pad(values, [(1, 0), (0, 0), (0, 0)]) - jnp.pad(values, [(0, 1), (0, 0), (0, 0)])
    def advect(values):
        return divergence(flux * jnp.where(flux >= 0., values[:-1], values[1:]))
    dice = phase + advect(ice_ratio)
    mass, dmass = gas + ice, dgas + dice
    momentum = jnp.stack([advect((1. + ice_ratio) * velocity) for velocity in wind])
    dvelocity = (momentum - wind * dmass[None]) / mass[None]
    kinetic = jnp.sum(wind**2, axis=0) / 2.
    energy = advect(hg + ice_ratio * hi + (1. + ice_ratio) * kinetic)
    rhs = (energy - hg * dgas - hi * dice - kinetic * dmass
           - mass * jnp.sum(wind * dvelocity, axis=0))
    dtemperature = _thermal_solve(rhs, capacity, ice, geometry)
    dice_ratio = (dice - ice_ratio * dgas) / gas
    return dtemperature, -body.gravity_m_s2 * total, dice_ratio, dvelocity


def settling_velocity(temperature, pressure, body, cf):
    """Stokes settling with Cunningham slip; viscosity follows CO2 Sutherland law."""
    viscosity = 1.370e-5 * (temperature / 273.)**1.5 * (273. + 222.) / (temperature + 222.)
    density = pressure / (body.gas_constant_j_kg_k * temperature)
    mean_free_path = viscosity / density * jnp.sqrt(jnp.pi / (2. * body.gas_constant_j_kg_k * temperature))
    knudsen = mean_free_path / cf.cloud_particle_radius_m
    slip = 1. + knudsen * (1.257 + .4 * jnp.exp(-1.1 / knudsen))
    return (2. * cf.cloud_ice_density_kg_m3 * cf.cloud_particle_radius_m**2
            * body.gravity_m_s2 / (9. * viscosity) * slip)


def settle(temperature, ps_pa, ice_ratio, wind, surface_temperature, sigma, dsigma, body, cf, dt):
    """Move suspended ice and its energy/momentum; return heat and ground impulse.

    Implicit fluxes use the updated donor inventory. No fall-speed/CFL cap is
    applied. Thermal equilibration and inelastic mixing return lost kinetic and
    gravitational energy as heat. The coupled heat solve includes changing
    hydrostatic cloud heights, rather than silently discarding that energy.
    """
    gas, ice, geometry, hg, hi, capacity = layer_properties(
        temperature, ps_pa, ice_ratio, sigma, dsigma, body, cf)
    pressure = jnp.asarray(sigma, dtype=ps_pa.dtype)[:, None, None] * ps_pa[None]
    thickness = gas * body.gas_constant_j_kg_k * temperature / pressure
    rate = settling_velocity(temperature, pressure, body, cf) / thickness
    def row(incoming, values):
        old, coefficient = values
        new = (old + incoming) / (1. + dt * coefficient)
        outgoing = dt * coefficient * new
        return outgoing, (new, outgoing)
    def transport(field):
        deposited, (remaining, _) = jax.lax.scan(row, jnp.zeros_like(field[0]), (field, rate))
        return remaining, deposited
    new_ice, deposit = transport(ice)
    old_mass, new_mass = gas + ice, gas + new_ice
    transported_momentum = [transport(ice * v) for v in wind]
    new_wind = jnp.stack([(gas * v + result[0]) / new_mass
                         for v, result in zip(wind, transported_momentum)])
    kinetic = jnp.sum(wind**2, axis=0) / 2.
    new_kinetic = jnp.sum(new_wind**2, axis=0) / 2.
    ice_energy, ground_energy = transport(ice * (hi + kinetic))
    energy = gas * (hg + kinetic) + ice_energy
    new_capacity = body.cp_j_kg_k * gas + cf.cloud_ice_heat_capacity_j_kg_k * new_ice
    new_temperature = _thermal_solve(
        energy + cf.latent_heat_j_kg * new_ice - new_mass * new_kinetic,
        new_capacity, new_ice, geometry)
    # Surface frost enthalpy changes with its mass; only the excess heats soil.
    frost_enthalpy = cf.cloud_ice_heat_capacity_j_kg_k * surface_temperature - cf.latent_heat_j_kg
    ground_heat = ground_energy - deposit * frost_enthalpy
    impulse = jnp.stack([result[1] for result in transported_momentum])
    return new_temperature, new_ice / gas, new_wind, deposit, ground_heat, impulse


def column_budgets(temperature, ps_pa, ice_ratio, wind, sigma, dsigma, body, cf):
    """Resolved column mass, moist enthalpy+potential+kinetic energy, momentum."""
    gas, ice, _, hg, hi, _ = layer_properties(temperature, ps_pa, ice_ratio, sigma, dsigma, body, cf)
    mass = gas + ice
    energy = jnp.sum(gas * hg + ice * hi + mass * jnp.sum(wind**2, axis=0) / 2., axis=0)
    return jnp.sum(mass, axis=0), energy, jnp.sum(mass[None] * wind, axis=1)
