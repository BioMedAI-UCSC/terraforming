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


def surface_exchange(temperature, ps_pa, ice_ratio, wind, surface_temperature,
                     frost_mass, deposit_rate, sigma, dsigma, body, cf):
    """Carry gas mass, energy and momentum between the lower air and frost.

    Ames mars_physics distributes surface exchange over its lowest pressure
    levels. Sigma-coordinate redistribution must carry the donor enthalpy,
    cloud mass and momentum. Sublimated gas enters at surface temperature and
    at rest relative to the ground. Deposition transfers air momentum to ground.
    """
    gas, ice, geometry, hg, hi, capacity = layer_properties(
        temperature, ps_pa, ice_ratio, sigma, dsigma, body, cf)
    sigma = np.asarray(sigma)
    below = np.flatnonzero(sigma / sigma[-1] < cf.surface_exchange_pressure_ratio)
    # Ames: klevs=kd-k+1; klev=kd-klevs-1 (Fortran one-based indices).
    first = max(0, int(below[-1]) - 2) if len(below) else 0
    weights = np.asarray(dsigma).copy()
    weights[:first] = 0.
    weights /= weights.sum()
    source = -deposit_rate[None] * jnp.asarray(weights)[:, None, None]
    dgas = -deposit_rate[None] * jnp.asarray(dsigma)[:, None, None]
    flux = jnp.cumsum(source - dgas, axis=0)[:-1]
    def advect(values):
        transfer = flux * jnp.where(flux >= 0., values[:-1], values[1:])
        return jnp.pad(transfer, [(1, 0), (0, 0), (0, 0)]) - jnp.pad(
            transfer, [(0, 1), (0, 0), (0, 0)])
    dice = advect(ice_ratio)
    dmass = dgas + dice
    mass = gas + ice
    source_wind = jnp.where(deposit_rate[None, None] >= 0., wind, 0.)
    momentum_source = source[None] * source_wind
    momentum = jnp.stack([advect((1. + ice_ratio) * v) for v in wind]) + momentum_source
    dvelocity = (momentum - wind * dmass[None]) / mass[None]
    kinetic = jnp.sum(wind**2, axis=0) / 2.
    source_energy = source * jnp.where(deposit_rate[None] >= 0.,
        hg + kinetic, body.cp_j_kg_k * surface_temperature[None])
    energy = advect(hg + ice_ratio * hi + (1. + ice_ratio) * kinetic) + source_energy
    rhs = (energy - hg * dgas - hi * dice - kinetic * dmass
           - mass * jnp.sum(wind * dvelocity, axis=0))
    dtemperature = _thermal_solve(rhs, capacity, ice, geometry)
    ground_energy = -jnp.sum(source_energy, axis=0)
    frost_enthalpy = cf.cloud_ice_heat_capacity_j_kg_k * surface_temperature - cf.latent_heat_j_kg
    surface_capacity = cf.thermal_inertia + frost_mass * cf.cloud_ice_heat_capacity_j_kg_k
    dsurface = (ground_energy - frost_enthalpy * deposit_rate) / surface_capacity
    impulse = -jnp.sum(momentum_source, axis=1)
    return dtemperature, (dice - ice_ratio * dgas) / gas, dvelocity, dsurface, impulse


def _remap_extensive(gas, target_gas, quantities):
    """Integrate donor-cell quantities onto new sigma-layer gas masses."""
    n, spatial = gas.shape[0], gas.shape[1:]
    count = quantities.shape[0]
    def column(donor, target, values):
        edges = jnp.concatenate([jnp.zeros(1, dtype=donor.dtype), jnp.cumsum(donor)])
        points = jnp.concatenate([jnp.zeros(1, dtype=target.dtype), jnp.cumsum(target)])
        index = jnp.clip(jnp.searchsorted(edges, points, side="right") - 1, 0, n - 1)
        integral = jnp.concatenate([jnp.zeros((count, 1), dtype=values.dtype), jnp.cumsum(values, axis=1)], axis=1)
        sampled = integral[:, index] + values[:, index] / donor[index] * (points - edges[index])[None]
        sampled = sampled.at[:, 0].set(0.).at[:, -1].set(jnp.sum(values, axis=1))
        return jnp.diff(sampled, axis=1)
    mapped = jax.vmap(column)(gas.reshape(n, -1).T, target_gas.reshape(n, -1).T,
        quantities.reshape(count, n, -1).transpose(2, 0, 1))
    return mapped.transpose(1, 2, 0).reshape((count, n) + spatial)


def saturation_adjustment(temperature, ps_pa, ice_ratio, wind, sigma, dsigma, body, cf):
    """Conservative post-dynamics CO2 saturation adjustment on the sigma grid.

    Ames uses the predicted temperature to form ice and release latent heat.
    Here each sweep remaps gas, ice, energy and momentum onto the changed sigma
    masses. The finite heat solve includes ice height and mixing dissipation.
    A transported budget limits net condensation to 10% of initial gas per step.
    """
    from src.framework.physics.gcm import co2_frost_point_k
    sigma_array = jnp.asarray(sigma)[:, None, None]
    weights = jnp.asarray(dsigma)[:, None, None]
    gas = ps_pa[None] * weights / body.gravity_m_s2
    initial = (temperature, ps_pa, ice_ratio, wind, .1 * gas)

    def sweep(state, _):
        t, ps, q, velocity, remaining = state
        gas, ice, geometry, hg, hi, capacity = layer_properties(t, ps, q, sigma, dsigma, body, cf)
        frost = co2_frost_point_k(sigma_array * ps[None])
        cold = (t < frost - 1.e-8) & (remaining > gas * 1.e-14)
        warm = (t > frost + 1.e-8) & (ice > gas * 1.e-14)
        def exchange(state):
            phi = jnp.einsum("ij,jxy->ixy", geometry, t)
            latent = cf.latent_heat_j_kg + (body.cp_j_kg_k - cf.cloud_ice_heat_capacity_j_kg_k) * frost - phi
            latent = jnp.where(latent > 0., latent, jnp.nan)
            diagonal_capacity = capacity + ice * jnp.diag(geometry)[:, None, None]
            phase = jnp.where(cold | warm, jnp.clip(
                diagonal_capacity * (frost - t) / latent, -ice, remaining), 0.)
            donor_gas = gas - phase
            new_ps = body.gravity_m_s2 * jnp.sum(donor_gas, axis=0)
            target_gas = new_ps[None] * weights / body.gravity_m_s2
            kinetic = jnp.sum(velocity**2, axis=0) / 2.
            mass = gas + ice
            energy = gas * hg + ice * hi + mass * kinetic
            quantities = jnp.concatenate([(ice + phase)[None], energy[None], mass[None] * velocity,
                                           (remaining - phase)[None]], axis=0)
            mapped = _remap_extensive(donor_gas, target_gas, quantities)
            new_ice, new_energy, momentum, new_budget = mapped[0], mapped[1], mapped[2:4], mapped[4]
            new_mass = target_gas + new_ice
            new_wind = momentum / new_mass[None]
            new_capacity = body.cp_j_kg_k * target_gas + cf.cloud_ice_heat_capacity_j_kg_k * new_ice
            new_temperature = _thermal_solve(new_energy + cf.latent_heat_j_kg * new_ice
                - new_mass * jnp.sum(new_wind**2, axis=0) / 2., new_capacity, new_ice, geometry)
            changed = jnp.any(phase != 0., axis=0)
            return (jnp.where(changed[None], new_temperature, t), jnp.where(changed, new_ps, ps),
                    jnp.where(changed[None], new_ice / target_gas, q),
                    jnp.where(changed[None, None], new_wind, velocity),
                    jnp.where(changed[None], new_budget, remaining))
        return jax.lax.cond(jnp.any(cold | warm), exchange, lambda s: s, state), None
    final, _ = jax.lax.scan(sweep, initial, None, length=12)
    return final[:4]


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
