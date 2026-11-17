"""Controlled column laboratory using the GCM radiation kernel and SI budgets.

This is a radiation/sensible-exchange experiment, not a replacement GCM. Pressure,
dust and illumination are fixed during each trajectory; there is no circulation,
regolith or CO2 phase change. The full GCM adapter is in radiation.py.
"""
from typing import NamedTuple
import math

from src.framework.gcm._dinosaur import jax, jnp
from .radiation import heating_rates


class ColumnState(NamedTuple):
    air_temperature_k: object
    surface_temperature_k: object
    external_energy_j_m2: object


def initial_column(inputs):
    return ColumnState(inputs.air_temperature_k, inputs.surface_temperature_k,
                       jnp.zeros_like(inputs.surface_temperature_k))


def column_energy(state, inputs, sigma_boundaries, body, surface_capacity):
    capacity = (body.cp_j_kg_k * inputs.surface_pressure_pa *
                jnp.diff(jnp.asarray(sigma_boundaries))[:, None, None] / body.gravity_m_s2)
    return (jnp.sum(capacity * state.air_temperature_k, axis=0)
            + surface_capacity * state.surface_temperature_k)


def make_column_step(inputs, sigma_boundaries, body, surface_capacity, flux_fn,
                     *, dt_seconds=30.0, exchange_fn=lambda params: 2.0):
    """Midpoint RK2 step(params, state) for short column experiments.

    flux_fn(params, ColumnInputs) supplies conventional or learned fluxes.
    exchange_fn(params) supplies sensible conductance in W/m²/K (bound it in
    the caller). External energy integrates the same midpoint TOA flux, enabling
    an exact discrete budget check up to roundoff for fixed layer masses.
    """
    if not (math.isfinite(dt_seconds) and math.isfinite(surface_capacity)) or dt_seconds <= 0 or surface_capacity <= 0:
        raise ValueError("positive timestep and surface heat capacity are required")
    lowest_capacity = (body.cp_j_kg_k * inputs.surface_pressure_pa *
                       (sigma_boundaries[-1] - sigma_boundaries[-2]) / body.gravity_m_s2)

    def tendency(params, state):
        current = inputs._replace(air_temperature_k=state.air_temperature_k,
                                  surface_temperature_k=state.surface_temperature_k)
        flux = flux_fn(params, current)
        air, surface = heating_rates(flux, inputs.surface_pressure_pa, sigma_boundaries,
                                     body, surface_capacity)
        sensible = exchange_fn(params) * (state.surface_temperature_k - state.air_temperature_k[-1])
        air = air.at[-1].add(sensible / lowest_capacity)
        surface = surface - sensible / surface_capacity
        return ColumnState(air, surface, flux.toa_net_down_w_m2)

    def step(params, state):
        k1 = tendency(params, state)
        middle = jax.tree.map(lambda x, dx: x + .5*dt_seconds*dx, state, k1)
        k2 = tendency(params, middle)
        return jax.tree.map(lambda x, dx: x + dt_seconds*dx, state, k2)
    return step
