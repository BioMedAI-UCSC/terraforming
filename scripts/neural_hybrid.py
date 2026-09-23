"""Mars adaptation of NeuralGCM column networks and differentiable rollouts."""
from __future__ import annotations
import dataclasses
import math
import numpy as np
from train_neural_pbl import Experiment, BODY
from src.framework.gcm._dinosaur import jax, jnp, spherical_harmonic, time_integration
from src.framework.gcm.dynamics import stepper, integrate
from src.framework.physics import gcm as physics
from src.framework.physics.neural_column import apply
from src.celestials.planets.mars.gcm import co2_forcing
from src.celestials.planets.mars.topography import regrid_to_nodal, mola_modal_orography


class HybridExperiment(Experiment):
    def __init__(self, max_dt=300., horizon=3, spinup=12, layers=12, *, mola=None,
                 flat_terrain=False, refresh_seconds=1800.):
        super().__init__(max_dt, horizon, spinup, layers)
        if not flat_terrain and mola is None:
            raise ValueError("MOLA terrain is required; --flat-terrain is for controlled tests")
        grid = self.coords.horizontal
        self.terrain = (np.zeros(grid.nodal_shape) if flat_terrain else regrid_to_nodal(self.coords, mola_path=mola))
        self.orography = mola_modal_orography(self.coords, self.specs, elevation_nodal_m=self.terrain)
        self.refresh_steps = max(1, int(refresh_seconds // self.dt))
        # Constant-coefficient implicit soil conduction, with observed past Ts
        # as boundary condition. No free-running atmospheric spinup.
        f = self.base
        cv = f.regolith_volumetric_heat_capacity_j_m3_k
        inertia = float(f.surface_thermal_inertia_tiu)
        dz = np.maximum(np.asarray(f.regolith_layer_skin_depth_fractions) * inertia / cv * math.sqrt(f.rotation_period_s / math.pi), 1e-4)
        k = inertia**2 / cv
        matrix = np.zeros((len(dz), len(dz)))
        boundary = np.zeros(len(dz)); boundary[0] = k / (.5 * dz[0]) / (cv * dz[0])
        matrix[0, 0] -= boundary[0]
        for i in range(len(dz) - 1):
            conductance = k / (.5 * (dz[i] + dz[i+1]))
            for a, b in ((i, i+1), (i+1, i)):
                rate = conductance / (cv * dz[a])
                matrix[a, a] -= rate; matrix[a, b] += rate
        self.soil_dt = self.interval / 12
        self.soil_inverse = jnp.asarray(np.linalg.inv(np.eye(len(dz)) - self.soil_dt * matrix))
        self.soil_boundary = jnp.asarray(boundary)[:, None, None]
        self.prepare_soil = jax.jit(self.soil_from_history)

    def soil_from_history(self, history):
        ground = jnp.broadcast_to(history[0], (len(self.soil_boundary), *history.shape[1:]))
        def interval(ground, pair):
            left, right = pair
            def substep(ground, fraction):
                ts = left + fraction * (right - left)
                rhs = ground + self.soil_dt * self.soil_boundary * ts
                return jnp.einsum("ij,jxy->ixy", self.soil_inverse, rhs), None
            return jax.lax.scan(substep, ground, jnp.arange(1, 13) / 12)[0], None
        return jax.lax.scan(interval, ground, (history[:-1], history[1:]))[0]

    def features(self, state, context):
        grid = self.coords.horizontal
        fields = self.fields(state)
        ps = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0] * self.pressure_scale
        pressure = jnp.asarray(self.coords.vertical.centers)[:, None, None] * ps
        dx, dy = grid.cos_lat_grad(state.dynamics.temperature_variation, clip=True)
        gradient = [grid.to_nodal(g) / (grid.cos_lat * BODY.radius_m) for g in (dx, dy)]
        levels = jnp.concatenate([*fields, grid.to_nodal(state.dynamics.vorticity) / self.time_scale,
                                  grid.to_nodal(state.dynamics.divergence) / self.time_scale,
                                  pressure, *gradient], axis=0)
        lon, sinlat = grid.nodal_mesh
        seconds = state.dynamics.sim_time * self.time_scale
        forcing = self.forcing(context)
        insolation = physics.cos_zenith_nodal(seconds, grid.latitudes, grid.longitudes, forcing) * physics.solar_flux(seconds, forcing)
        surface = jnp.stack([state.surface_temperature[0], state.ground_temperature[0],
                             state.co2_ice[0] * self.pressure_scale / BODY.gravity_m_s2,
                             context[1], insolation, sinlat, jnp.cos(lon), jnp.sin(lon), jnp.asarray(self.terrain)])
        return jnp.moveaxis(jnp.concatenate([levels, surface], axis=0), 0, -1)

    def output(self, network, state, context, stats, tendency=False):
        x = (self.features(state, context) - stats["mean"]) / stats["scale"]
        raw = apply(network, x)
        result = jnp.moveaxis(raw, -1, 0).reshape((3, self.coords.vertical.layers, *self.coords.horizontal.nodal_shape))
        scale = stats["tendency_scale" if tendency else "field_scale"]
        return .01 * result * scale[:, :, None, None]

    def modal_increment(self, fields):
        grid = self.coords.horizontal
        vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(grid, fields[1] / self.velocity_scale, fields[2] / self.velocity_scale)
        return vor, div, grid.to_modal(fields[0])

    def correct_state(self, state, increment):
        vor, div, heat = self.modal_increment(increment)
        d = state.dynamics
        return state._replace(dynamics=dataclasses.replace(d, vorticity=d.vorticity + vor,
                              divergence=d.divergence + div, temperature_variation=d.temperature_variation + heat))

    def advance(self, state, context, closure, intervals):
        # closure is (three-network parameter tree, frozen training statistics).
        def block(state, nsteps):
            residual = None
            if closure is not None:
                params, stats = closure
                rates = self.output(params["physics"], state, context, stats, tendency=True)
                residual = tuple(x * self.time_scale for x in self.modal_increment(rates))
            ode = physics.forced_co2_primitive_equations(self.coords, BODY, self.forcing(context), co2_forcing(),
                    specs=self.specs, orography=self.orography, neural_tendencies=residual)
            step = stepper(ode, self.dt, self.specs)
            diffusion = time_integration.horizontal_diffusion_step_filter(self.coords.horizontal,
                        self.dt / self.time_scale, .1 * self.base.rotation_period_s / self.time_scale, order=4)
            def filtered(state):
                result = step(state)
                return result._replace(dynamics=diffusion(state.dynamics, result.dynamics))
            safe = physics.positivity_preserving_co2_step(filtered, self.coords, self.specs)
            return integrate(jax.checkpoint(safe), state, nsteps)
        def interval(state, _):
            full, remainder = divmod(self.steps, self.refresh_steps)
            def refresh(state, _):
                return block(state, self.refresh_steps), None
            state = jax.lax.scan(jax.checkpoint(refresh), state, None, length=full)[0]
            if remainder:
                state = block(state, remainder)
            return state, None
        return jax.lax.scan(jax.checkpoint(interval), state, None, length=intervals)[0]

    def forecast(self, params, example, stats, horizon, *, physical=False, return_latent=False):
        state, context, _, _ = example
        if not physical:
            state = self.correct_state(state, self.output(params["encoder"], state, context, stats))
        def interval(state, _):
            state = self.advance(state, context, None if physical else (params, stats), 1)
            decoded = state if physical else self.correct_state(state, self.output(params["decoder"], state, context, stats))
            return state, (self.fields(decoded), self.fields(state))
        decoded, latent = jax.lax.scan(jax.checkpoint(interval), state, None, length=horizon)[1]
        return (decoded, latent) if return_latent else decoded

    def metrics(self, params, example, stats, horizon, *, physical=False, return_forecasts=False):
        state, context, target, mass = example
        prediction, latent = self.forecast(params, example, stats, horizon, physical=physical, return_latent=True)
        target, mass = target[:horizon], mass[:horizon]
        error = (prediction - target) / stats["field_scale"][None, :, :, None, None]
        weights = mass[:, None]
        mse = jnp.sum(error**2 * weights) / (3 * jnp.sum(mass))
        # Large-scale bias term; data MSE already measures all resolved scales.
        mean_error = jnp.sum(error * weights, axis=(-2, -1)) / jnp.sum(weights, axis=(-2, -1))
        bias = jnp.mean(mean_error**2)
        latent_error = (latent - target) / stats["field_scale"][None, :, :, None, None]
        latent_mse = jnp.sum(latent_error**2 * weights) / (3 * jnp.sum(mass))
        rmse = jnp.sqrt(jnp.sum((prediction-target)**2 * weights, axis=(0,3,4)) / jnp.sum(weights, axis=(0,3,4)))
        metrics = (mse, bias, latent_mse, rmse)
        return (*metrics, prediction, latent) if return_forecasts else metrics

    def loss(self, params, example, stats, horizon, regularization=1e-4):
        mse, bias, latent_mse, _ = self.metrics(params, example, stats, horizon)
        state, context, _, _ = example
        encoded = self.correct_state(state, self.output(params["encoder"], state, context, stats))
        roundtrip = self.correct_state(encoded, self.output(params["decoder"], encoded, context, stats))
        reconstruction = jnp.mean(((self.fields(roundtrip) - self.fields(state)) / stats["field_scale"][:, :, None, None])**2)
        # Scaled output penalties, including adapters, prevent cost-free drift.
        penalty = sum(jnp.mean((self.output(params[name], state, context, stats, tendency=name == "physics") /
                         stats["tendency_scale" if name == "physics" else "field_scale"][:, :, None, None])**2)
                      for name in ("encoder", "physics", "decoder"))
        return mse + .5 * latent_mse + .1 * bias + reconstruction + regularization * penalty


def examples(experiment, ds, horizon, skip=0):
    grid = experiment.coords.horizontal
    area = np.asarray(grid.quadrature_weights)
    dsigma = np.diff(experiment.coords.vertical.boundaries)[:, None, None]
    for offset in range(skip, ds.sizes["time"] - experiment.spinup - horizon):
        i = offset + experiment.spinup
        state, context = experiment.snapshot(ds, i)
        history = jnp.asarray(ds.tsurf.isel(time=slice(offset, i+1)).transpose("time", "lon", "lat").values)
        state = state._replace(ground_temperature=experiment.prepare_soil(history))
        selection = ds.isel(time=slice(i+1, i+1+horizon))
        target = np.stack([selection[name].transpose("time", "lev", "lon", "lat").values for name in ("temp", "uwind", "vwind")], axis=1)
        mass = selection.psurf.transpose("time", "lon", "lat").values[:, None] / BODY.gravity_m_s2 * dsigma * area
        yield jax.device_get((state, context, target, mass))


def fit_statistics(experiment, datasets):
    """Caller supplies training chunks only; never update after initialization."""
    total = squares = None
    count = 0
    fields, tendencies = [], []
    for ds in datasets:
        for i in range(experiment.spinup, ds.sizes["time"] - 1, 12):
            state, context = experiment.snapshot(ds, i)
            history = jnp.asarray(ds.tsurf.isel(time=slice(i-experiment.spinup, i+1)).transpose("time", "lon", "lat").values)
            state = state._replace(ground_temperature=experiment.prepare_soil(history))
            x = np.asarray(experiment.features(state, context)).reshape(-1, 8 * experiment.coords.vertical.layers + 9)
            total = x.sum(0) if total is None else total + x.sum(0)
            squares = (x*x).sum(0) if squares is None else squares + (x*x).sum(0)
            count += len(x)
            pair = np.stack([ds[name].isel(time=slice(i, i+2)).transpose("time", "lev", "lon", "lat").values for name in ("temp", "uwind", "vwind")], axis=1)
            fields.append(pair[0]); tendencies.append((pair[1]-pair[0]) / experiment.interval)
    if not count:
        raise ValueError("no training snapshots for normalization")
    mean = total / count
    stats = dict(mean=mean, scale=np.maximum(np.sqrt(np.maximum(squares/count - mean**2, 0)), 1e-6),
                 field_scale=np.maximum(np.std(np.stack(fields), axis=(0, 3, 4)), 1.),
                 tendency_scale=np.maximum(np.std(np.stack(tendencies), axis=(0, 3, 4)), 1e-8))
    if not all(np.isfinite(value).all() for value in stats.values()):
        raise ValueError("nonfinite training normalization")
    return stats
