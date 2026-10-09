"""Column features and flux-conserving radiation replacement for the GCM."""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.framework.gcm._dinosaur import jnp, jax, scales
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.gcm import (
    column_radiative_fluxes, radiative_flux_diagnostics, dust_optical_depths,
    cos_zenith_nodal, solar_path_cosine_nodal, solar_flux,
)
from .models import ColumnMLP


class ColumnInputs(NamedTuple):
    """SI fields; air is (layer, x, y); other fields broadcast to (x, y)."""
    air_temperature_k: object
    surface_temperature_k: object
    surface_pressure_pa: object
    dust_visible: object
    dust_longwave: object
    dust_top_km: object
    incoming_solar_w_m2: object
    solar_path_factor: object
    albedo: object
    emissivity: object


def feature_schema(n_layers):
    return ([f"air_temperature_k[{k}]" for k in range(n_layers)]
            + [f"pressure_mid_pa[{k}]" for k in range(n_layers)]
            + list(ColumnInputs._fields[1:]))


def column_features(inputs, sigma_boundaries):
    """Return (x, y, 2*layers+9), with layers ordered TOA to surface."""
    air = inputs.air_temperature_k
    if air.ndim != 3 or len(sigma_boundaries) != air.shape[0] + 1:
        raise ValueError("expected (layer, x, y) temperatures and matching sigma boundaries")
    sigma = jnp.asarray(sigma_boundaries)
    mid = (sigma[1:] + sigma[:-1]) * 0.5
    ps = jnp.broadcast_to(jnp.asarray(inputs.surface_pressure_pa), air.shape[1:])
    pressure = mid[:, None, None] * ps[None]
    scalar_fields = [jnp.broadcast_to(jnp.asarray(v), air.shape[1:]) for v in inputs[1:]]
    return jnp.concatenate([jnp.moveaxis(air, 0, -1), jnp.moveaxis(pressure, 0, -1),
                            jnp.stack(scalar_fields, axis=-1)], axis=-1)


def inputs_from_state(state, coords, specs, body, forcing, *,
                      air_temperature_k=None, surface_pressure_pa=None):
    """Extract inference-only inputs; no reference fluxes or future state needed."""
    grid = coords.horizontal
    if air_temperature_k is None:
        air_temperature_k = (grid.to_nodal(state.dynamics.temperature_variation)
                             + reference_temperature(coords, body)[:, None, None])
    if surface_pressure_pa is None:
        surface_pressure_pa = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0] * float(
            specs.dimensionalize(1.0, scales.units.pascal).magnitude)
    seconds = state.dynamics.sim_time / float(specs.nondimensionalize(1.0 * scales.units.second))
    cz = cos_zenith_nodal(seconds, grid.latitudes, grid.longitudes, forcing)
    path = (1 / jnp.clip(solar_path_cosine_nodal(seconds, grid.latitudes, grid.longitudes, forcing), .05, 1)
            if forcing.solar_slant_path_enabled else jnp.ones_like(cz))
    visible, longwave = dust_optical_depths(seconds, forcing)
    return ColumnInputs(jnp.clip(air_temperature_k, 1, None),
                        jnp.clip(state.surface_temperature[0], 1, None), surface_pressure_pa,
                        visible, longwave, forcing.dust_top_height_km,
                        solar_flux(seconds, forcing) * cz, path, forcing.albedo, forcing.emissivity)


def reference_fluxes(inputs, sigma_boundaries, body, forcing):
    """Evaluate the same conventional radiation kernel used by the full GCM."""
    import dataclasses
    f = dataclasses.replace(forcing, albedo=inputs.albedo, emissivity=inputs.emissivity,
                            dust_top_height_km=inputs.dust_top_km)
    return column_radiative_fluxes(
        inputs.air_temperature_k, inputs.surface_temperature_k, inputs.surface_pressure_pa,
        inputs.incoming_solar_w_m2, inputs.solar_path_factor, inputs.dust_visible,
        inputs.dust_longwave, sigma_boundaries, body, f)


@dataclass(frozen=True)
class NeuralRadiation:
    """JAX-compatible model adapter. A custom apply(params, features) may replace MLP.

    apply returns feature-last 4*(layers+1) raw outputs. SW fluxes are bounded
    fractions of incoming sunlight; LW fluxes use positive softplus outputs.
    Boundary conditions are imposed exactly, not learned. The reference compact
    solver omits upward atmospheric SW, so its emulator does too when configured.
    """
    sigma_boundaries: tuple[float, ...]
    stefan_boltzmann: float = 5.670374419e-8
    hidden_sizes: tuple[int, ...] = (32, 32)
    compact_shortwave: bool = False

    def __post_init__(self):
        b = np.asarray(self.sigma_boundaries)
        if len(b) < 2 or b[0] != 0 or b[-1] != 1 or not np.all(np.diff(b) > 0):
            raise ValueError("sigma boundaries must increase from 0 to 1")

    @property
    def model(self):
        n = len(self.sigma_boundaries) - 1
        return ColumnMLP(2*n + 9, 4*(n+1), self.hidden_sizes)

    def init(self, key):
        return self.model.init(key)

    def predict(self, params, inputs, normalization, *, apply=None):
        x = normalization.apply(column_features(inputs, self.sigma_boundaries))
        raw = (self.model.apply if apply is None else apply)(params, x)
        n = len(self.sigma_boundaries)
        if raw.shape != x.shape[:-1] + (4*n,):
            raise ValueError("radiation model output shape mismatch")
        raw = jnp.moveaxis(raw.reshape(x.shape[:-1] + (4, n)), (-2, -1), (0, 1))
        incoming = jnp.maximum(jnp.asarray(inputs.incoming_solar_w_m2), 0)
        sw = jax.nn.sigmoid(raw[0]) * incoming
        sw = sw.at[0].set(jnp.broadcast_to(incoming, sw.shape[1:]))
        sw_up = (jnp.zeros_like(sw) if self.compact_shortwave
                 else jax.nn.sigmoid(raw[1]) * incoming)
        sw_up = sw_up.at[-1].set(inputs.albedo * sw[-1])
        lw_up = jax.nn.softplus(raw[2]) * 100.0
        lw_up = lw_up.at[-1].set(jnp.broadcast_to(
            inputs.emissivity * self.stefan_boltzmann * inputs.surface_temperature_k**4,
            lw_up.shape[1:]))
        lw_down = (jax.nn.softplus(raw[3]) * 100.0).at[0].set(0)
        return radiative_flux_diagnostics(sw, sw_up, lw_up, lw_down)

    def bind(self, params, normalization, *, apply=None):
        """Return a drop-in radiation_component for forced[_co2]_primitive_equations."""
        def component(state, coords, specs, body, forcing, **kwargs):
            if tuple(np.asarray(coords.vertical.boundaries)) != self.sigma_boundaries:
                raise ValueError("radiation model sigma configuration mismatch")
            if forcing.stefan_boltzmann != self.stefan_boltzmann:
                raise ValueError("radiation model Stefan-Boltzmann constant mismatch")
            if self.compact_shortwave == forcing.ames_correlated_k_enabled:
                raise ValueError("radiation model reference scheme mismatch")
            inputs = inputs_from_state(state, coords, specs, body, forcing, **kwargs)
            return self.predict(params, inputs, normalization, apply=apply)
        return component


def radiation_budget_residual(fluxes):
    """W/m² closure residual; zero means exchanges telescope, not perfect physics."""
    return (jnp.sum(fluxes.atmospheric_convergence_w_m2, axis=0)
            + fluxes.surface_net_w_m2 - fluxes.toa_net_down_w_m2)


def heating_rates(fluxes, surface_pressure_pa, sigma_boundaries, body, surface_capacity):
    """Return atmospheric and surface radiation-only heating in K/s."""
    capacity = (body.cp_j_kg_k * surface_pressure_pa
                * jnp.diff(jnp.asarray(sigma_boundaries))[:, None, None] / body.gravity_m_s2)
    return (fluxes.atmospheric_convergence_w_m2 / jnp.maximum(capacity, 1e-6),
            fluxes.surface_net_w_m2 / surface_capacity)
