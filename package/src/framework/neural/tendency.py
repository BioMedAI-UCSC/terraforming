"""Bounded neural residual tendencies for differentiable planetary experiments."""
from dataclasses import dataclass

import numpy as np

from src.framework.gcm._dinosaur import jnp, scales, spherical_harmonic
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.gcm import ColumnPhysicsTendencies, cos_zenith_nodal, solar_flux
from .models import ColumnMLP


def state_feature_schema(n_layers):
    """Names for the feature-last state vector consumed by a policy."""
    return ([f"air_temperature_k[{i}]" for i in range(n_layers)]
            + ["surface_temperature_k", "surface_pressure_pa", "eastward_wind_ms",
               "northward_wind_ms", "sin_latitude", "cos_latitude", "sin_longitude",
               "cos_longitude", "insolation_w_m2"])


def state_features(state, coords, specs, body, forcing):
    """Build local, current-state-only policy inputs with shape ``(lon, lat, features)``."""
    grid = coords.horizontal
    n = coords.vertical.layers
    air = grid.to_nodal(state.dynamics.temperature_variation) + jnp.asarray(
        reference_temperature(coords, body))[:, None, None]
    surface = state.surface_temperature[0]
    pressure = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0] * float(
        specs.dimensionalize(1.0, scales.units.pascal).magnitude)
    wind = spherical_harmonic.vor_div_to_uv_nodal(
        grid, state.dynamics.vorticity, state.dynamics.divergence)
    velocity_unit = scales.units.meter / scales.units.second
    u = specs.dimensionalize(wind[0], velocity_unit).magnitude[-1]
    v = specs.dimensionalize(wind[1], velocity_unit).magnitude[-1]
    lat = jnp.asarray(grid.latitudes)[None, :]
    lon = jnp.asarray(grid.longitudes)[:, None]
    seconds = state.dynamics.sim_time / float(specs.nondimensionalize(1.0 * scales.units.second))
    insolation = solar_flux(seconds, forcing) * cos_zenith_nodal(
        seconds, grid.latitudes, grid.longitudes, forcing)
    return jnp.concatenate([
        jnp.moveaxis(air, 0, -1), surface[..., None], pressure[..., None],
        u[..., None], v[..., None], jnp.broadcast_to(jnp.sin(lat), surface.shape)[..., None],
        jnp.broadcast_to(jnp.cos(lat), surface.shape)[..., None],
        jnp.broadcast_to(jnp.sin(lon), surface.shape)[..., None],
        jnp.broadcast_to(jnp.cos(lon), surface.shape)[..., None], insolation[..., None],
    ], axis=-1)


@dataclass(frozen=True)
class NeuralTendency:
    """A bounded neural correction to one prognostic physical tendency.

    The reference implementation supports temperature heating in kelvin/second.
    The model output is passed through tanh, so the configured bound is enforced
    before conversion to the framework's nondimensional tendency units. The
    adapter returns zero for all other tendencies and leaves surface reservoirs
    under conventional physics control.
    """
    n_layers: int
    maximum_heating_k_s: float = 2.0e-4
    hidden_sizes: tuple[int, ...] = (32, 32)
    output: str = "temperature_heating"

    def __post_init__(self):
        if self.n_layers < 1 or not np.isfinite(self.maximum_heating_k_s) or self.maximum_heating_k_s <= 0:
            raise ValueError("n_layers and maximum_heating_k_s must be positive")
        if self.output != "temperature_heating":
            raise ValueError("only temperature_heating is currently supported")

    @property
    def model(self):
        return ColumnMLP(len(state_feature_schema(self.n_layers)), 1, self.hidden_sizes)

    def init(self, key):
        return self.model.init(key)

    def bind(self, params, normalization, *, coords=None, specs=None, body=None,
             forcing=None, apply=None):
        """Bind a state-only callable for ``forced_primitive_equations``.

        The simulation context is static and therefore supplied at binding time;
        trainable arrays remain in ``params`` and are traced by JAX.
        """
        if any(value is None for value in (coords, specs, body, forcing)):
            raise ValueError("coords, specs, body and forcing are required when binding a tendency")

        def component(state):
            if coords.vertical.layers != self.n_layers:
                raise ValueError("neural tendency layer count does not match coordinates")
            features = state_features(state, coords, specs, body, forcing)
            if features.shape[-1] != normalization.mean.shape[-1]:
                raise ValueError("neural tendency normalization feature count mismatch")
            raw = (self.model.apply if apply is None else apply)(params, normalization.apply(features))
            heating_si = self.maximum_heating_k_s * jnp.tanh(raw[..., 0])
            heating_nd = specs.nondimensionalize(heating_si * scales.units.kelvin / scales.units.second)
            heat_modal = coords.horizontal.to_modal(
                jnp.broadcast_to(heating_nd, (self.n_layers,) + features.shape[:2]))
            zeros = jnp.zeros_like(heat_modal)
            return ColumnPhysicsTendencies(
                zeros, zeros, heat_modal, jnp.zeros_like(state.dynamics.log_surface_pressure),
                jnp.zeros_like(state.surface_temperature), jnp.zeros_like(state.co2_ice),
                jnp.zeros_like(state.ground_temperature),
                {name: jnp.zeros_like(value) for name, value in state.dynamics.tracers.items()},
            )
        return component
