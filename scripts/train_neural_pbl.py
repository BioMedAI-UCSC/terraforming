#!/usr/bin/env python3
"""Train one neural PBL model with synchronous JAX data parallelism on one host."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import pickle
import time
import sys

from neural_logging import LOGGER, configure, event, phase
from itertools import islice

import numpy as np

from neural_macda import NativeCache, atomic_bytes, locked, windows
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import integrate, reference_temperature, stepper
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics
from src.framework.physics.neural_pbl import Closure, initialize, multipliers, FEATURE_NAMES
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing, co2_forcing
from src.celestials.planets.mars.maps import initial_rest_state


class Experiment:
    def __init__(self, max_dt=300., horizon=1, spinup=12, layers=12):
        self.coords = coordinate_system("T21", layers)
        self.specs = physics_specs(BODY)
        self.base = dataclasses.replace(
            radiative_forcing(diurnal=True, co2_radiation_enabled=True),
            pbl_diffusion_enabled=True, regolith_enabled=True,
            stability_exchange_enabled=True, convective_adjustment_enabled=True,
        )
        self.interval = self.base.rotation_period_s / 12
        self.steps = math.ceil(self.interval / max_dt)
        self.dt = self.interval / self.steps
        self.horizon, self.spinup = horizon, spinup
        u = scales.units
        self.velocity_scale = float(self.specs.dimensionalize(1., u.meter / u.second).magnitude)
        self.pressure_scale = float(self.specs.dimensionalize(1., u.pascal).magnitude)
        self.time_scale = float(self.specs.dimensionalize(1., u.second).magnitude)
        self.ref = jnp.asarray(reference_temperature(self.coords, BODY))[:, None, None]

    def forcing(self, context):
        angle, dust = context
        return dataclasses.replace(self.base, init_orbital_angle_rad=angle,
                                   dust_visible_optical_depth=dust,
                                   dust_longwave_optical_depth=dust / 3.,
                                   pbl_implicit_timestep_s=self.dt)

    def advance(self, state, context, closure, intervals):
        if intervals == 0:
            return state
        ode = physics.forced_co2_primitive_equations(
            self.coords, BODY, self.forcing(context), co2_forcing(),
            specs=self.specs, neural_closure=closure,
        )
        step = physics.positivity_preserving_co2_step(
            stepper(ode, self.dt, self.specs), self.coords, self.specs)
        return integrate(jax.checkpoint(step), state, self.steps * intervals)

    def fields(self, state):
        grid = self.coords.horizontal
        u, v = spherical_harmonic.vor_div_to_uv_nodal(grid, state.dynamics.vorticity, state.dynamics.divergence)
        return jnp.stack((grid.to_nodal(state.dynamics.temperature_variation) + self.ref,
                          u * self.velocity_scale, v * self.velocity_scale))

    def snapshot(self, ds, i):
        grid = self.coords.horizontal
        def field(name):
            a = ds[name].isel(time=i)
            return jnp.asarray(a.transpose(*(["lev"] if "lev" in a.dims else []), "lon", "lat").values)
        t, u, v = (field(name) for name in ("temp", "uwind", "vwind"))
        vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(
            grid, u / self.velocity_scale, v / self.velocity_scale)
        seconds = float(ds.time[i]) * self.base.rotation_period_s
        dynamics = dataclasses.replace(
            initial_rest_state(self.coords, self.specs, BODY, np.zeros(grid.nodal_shape)),
            temperature_variation=grid.to_modal(t - self.ref), vorticity=vor, divergence=div,
            log_surface_pressure=grid.to_modal(jnp.log(field("psurf")[None] / self.pressure_scale)),
            sim_time=jnp.asarray(seconds / self.time_scale),
        )
        state = physics.initial_column_state(dynamics, self.coords, 200., self.specs, forcing=self.base)
        # Isothermal soil at observed surface temperature, followed by a shared
        # physical warmup. Neither branch receives future soil/atmosphere data.
        state = state._replace(surface_temperature=field("tsurf")[None],
                               co2_ice=field("co2ice")[None] * BODY.gravity_m_s2 / self.pressure_scale,
                               ground_temperature=jnp.broadcast_to(field("tsurf"), state.ground_temperature.shape))
        angle = physics.mean_anomaly_for_ls(math.radians(float(ds.Ls[i])), self.base)
        angle -= 2 * math.pi * seconds / self.base.orbital_period_s
        return state, (jnp.asarray(angle), field("coldust"))

    def features(self, state, context):
        values = []
        physics.pbl_vertical_diffusion_tendencies(
            state, self.coords, self.specs, BODY, self.forcing(context), feature_sink=values.append)
        return jnp.stack(values)

    def loss(self, params, example, mean, scale, regularization):
        state, context, target, mass, features = example
        closure = Closure(params, mean, scale)
        def interval(carry, _):
            result = self.advance(carry, context, closure, 1)
            return result, self.fields(result)
        _, predicted = jax.lax.scan(interval, state, None, length=self.horizon)
        # Fixed physical scales give T and the two winds comparable influence.
        error = (predicted - target) / jnp.asarray([10., 10., 10.])[None, :, None, None, None]
        forecast = jnp.sum(error**2 * mass[:, None]) / (3 * jnp.sum(mass))
        penalty = jnp.mean(jnp.log(multipliers(closure, features))**2)
        return forecast + regularization * penalty


def synchronized_update(loss_fn, learning_rate, devices):
    """Average gradients BEFORE Adam; all replicas own the same optimizer state."""
    def update(params, first, second, count, batch):
        value, gradient = jax.value_and_grad(loss_fn)(params, batch)
        gradient = jax.lax.pmean(gradient, "devices")
        value = jax.lax.pmean(value, "devices")
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(gradient)))
        gradient = jax.tree_util.tree_map(lambda g: g / jnp.maximum(1., norm), gradient)
        count = count + 1
        first = jax.tree_util.tree_map(lambda m, g: .9 * m + .1 * g, first, gradient)
        second = jax.tree_util.tree_map(lambda v, g: .999 * v + .001 * g**2, second, gradient)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - learning_rate * (m / (1 - .9**count)) / (jnp.sqrt(v / (1 - .999**count)) + 1e-8),
            params, first, second)
        return params, first, second, count, value, norm
    return jax.pmap(update, axis_name="devices", devices=devices)


def device_batch(examples, devices):
    """Place one example/replica on each selected device explicitly."""
    if len(examples) != len(devices):
        raise ValueError("batch must contain one example per device")
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(devices), ("devices",)),
        jax.sharding.PartitionSpec("devices"),
    )
    return jax.tree_util.tree_map(
        lambda *xs: jax.device_put(np.stack([np.asarray(x) for x in xs]), sharding),
        *examples,
    )


def parallel_prepare(experiment, devices):
    """Run each trajectory's physical warmup on its own GPU, outside AD."""
    def prepare(example):
        state, context, target, mass, _ = example
        state = experiment.advance(state, context, None, experiment.spinup)
        return state, context, target, mass, experiment.features(state, context)
    return jax.pmap(prepare, devices=devices)


def verify_device_placement(tree, devices, stage):
    expected = set(devices)
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf.devices() != expected:
            raise RuntimeError(f"{stage} is not distributed over all selected devices")
    event("device_placement_verified", stage=stage, devices=[str(d) for d in devices])


def examples(experiment, ds, warmup, skip=0, *, defer_warmup=False):
    grid = experiment.coords.horizontal
    area = jnp.asarray(grid.quadrature_weights)
    dsigma = jnp.asarray(np.diff(experiment.coords.vertical.boundaries))[:, None, None]
    for i in range(skip, ds.sizes["time"] - experiment.spinup - experiment.horizon):
        state, context = experiment.snapshot(ds, i)
        if not defer_warmup:
            with phase("example_prepare", snapshot=i, spinup_intervals=experiment.spinup):
                state = jax.block_until_ready(warmup(state, context))
        start = i + experiment.spinup + 1
        selection = ds.isel(time=slice(start, start + experiment.horizon))
        target = jnp.stack([jnp.asarray(selection[name].transpose("time", "lev", "lon", "lat").values)
                            for name in ("temp", "uwind", "vwind")], axis=1)
        ps = jnp.asarray(selection.psurf.transpose("time", "lon", "lat").values)
        mass = ps[:, None] / BODY.gravity_m_s2 * dsigma * area
        features = (jnp.zeros((experiment.coords.vertical.layers - 1, *grid.nodal_shape, 13))
                    if defer_warmup else experiment.features(state, context))
        yield state, context, target, mass, features


def validation_subset(cache, experiment, warmup, count):
    """Fixed small holdout spanning the validation chronology, kept on the host."""
    selections = windows("validation")
    chosen = [selections[i] for i in np.linspace(0, len(selections) - 2, min(2, count), dtype=int)]
    subset = []
    for index, ds in enumerate(cache.iterate(chosen, "validation")):
        needed = (count - len(subset) + len(chosen) - index - 1) // (len(chosen) - index)
        subset.extend(jax.device_get(example) for example in
                      islice(examples(experiment, ds, warmup), needed))
    if len(subset) != count:
        raise ValueError("validation chunks cannot supply the requested example count")
    return subset, chosen


def validation_report(params, subset, evaluate, physical_scores, step, steps_per_epoch):
    scores = []
    for index, example in enumerate(subset, 1):
        with phase("validation_example", step=step, example=index, total=len(subset)):
            scores.append(float(evaluate(params, example)))
        event("validation_progress", step=step, completed=index, total=len(subset), neural_loss=scores[-1])
    scores = np.asarray(scores)
    if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(physical_scores).all():
        raise ValueError("empty/nonfinite small validation; training stopped")
    return dict(step=step, epoch=step / steps_per_epoch, split="validation",
                examples=len(scores), neural_loss=float(scores.mean()),
                physical_loss=float(np.mean(physical_scores)))


def validation_due(step, steps_per_epoch, every_epochs, every_steps, final_step):
    return (step == final_step or
            (every_steps > 0 and step % every_steps == 0) or
            (every_epochs > 0 and step % (every_epochs * steps_per_epoch) == 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True)
    p.add_argument("--cache", default="data/neural_macda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="allow CPU devices for smoke tests")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--regularization", type=float, default=1e-3)
    p.add_argument("--max-dt", type=float, default=300.)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--spinup", type=int, default=12)
    p.add_argument("--normalization-chunks", type=int, default=7)
    p.add_argument("--constant", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", choices=("validation", "test", "storm"))
    p.add_argument("--stage", choices=("train", "validation", "test", "storm"))
    p.add_argument("--chunks", type=int, help="bound chunk count (also useful for a training smoke test)")
    p.add_argument("--preflight-only", action="store_true", help="small held-out physical/neural check without optimizer updates")
    p.add_argument("--validation-examples", type=int, default=4)
    p.add_argument("--validate-every-epochs", type=int, default=1, help="0 disables epoch-based checks")
    p.add_argument("--validate-every-steps", type=int, default=0, help="optional additional update-based checks")
    args = p.parse_args()
    if args.validation_examples < 1 or min(args.validate_every_epochs, args.validate_every_steps) < 0:
        p.error("validation examples must be positive and intervals nonnegative")
    if args.preflight_only and (args.evaluate or args.stage):
        p.error("preflight-only cannot be combined with evaluate or stage")
    if min(args.devices, args.steps, args.horizon, args.normalization_chunks) < 1 or args.spinup < 0 or args.max_dt <= 0:
        p.error("invalid device/step/horizon/spinup/normalization/timestep count")
    if args.chunks is not None and args.chunks < 1:
        p.error("chunks must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.regularization) or args.regularization < 0:
        p.error("learning rate must be positive and regularization nonnegative")
    if args.horizon + args.spinup >= 120:
        p.error("spinup plus horizon must be less than the 120-snapshot chunk")
    configure(args.output)
    event("run_start", arguments=vars(args), python=sys.version, executable=sys.executable, jax_version=jax.__version__)
    jax.config.update("jax_enable_x64", True)
    cache = NativeCache(args.cache, args.revision)
    if args.stage:
        for ds in cache.iterate(windows(args.stage)[:args.chunks], args.stage):
            event("staging_progress", first_sol=float(ds.time[0]), last_sol=float(ds.time[-1]))
        return
    devices = jax.local_devices()[:args.devices]
    event("devices_selected", devices=[str(d) for d in devices], global_batch_size=len(devices),
          parallel_stages=["parallel_warmup", "training_update"],
          single_device_stages=["normalization", "validation", "snapshot_construction"])
    if len(devices) != args.devices or (not args.cpu and any(d.platform != "gpu" for d in devices)):
        p.error(f"requested {args.devices} GPUs; available devices: {jax.local_devices()}")
    args.output.mkdir(parents=True, exist_ok=True)
    with locked(args.output / "run.lock"):
        run(args, cache, devices)


def run(args, cache, devices):
    experiment = Experiment(args.max_dt, args.horizon, args.spinup)
    warmup = jax.jit(lambda state, context: experiment.advance(state, context, None, args.spinup))
    contract = {k: getattr(args, k) for k in ("revision", "seed", "constant", "max_dt", "horizon", "spinup",
                                            "normalization_chunks", "learning_rate", "regularization", "devices", "chunks")}
    contract.update(features=list(FEATURE_NAMES), version=1)
    checkpoint = args.output / "checkpoint.pkl"
    if args.resume or args.evaluate:
        # Only load checkpoints produced locally by this training command.
        event("checkpoint_loading", path=str(checkpoint))
        saved = pickle.loads(checkpoint.read_bytes())
        if args.evaluate:
            contract["chunks"] = saved["contract"]["chunks"]
        if saved["contract"] != contract:
            raise ValueError("checkpoint training/data contract differs from command")
        params, first, second = saved["params"], saved["first"], saved["second"]
        mean, scale = saved["mean"], saved["scale"]
        completed, position = saved["completed"], saved["position"]
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng"]
        event("checkpoint_loaded", step=completed, position=position, normalization="frozen")
    else:
        if checkpoint.exists():
            raise ValueError("output already contains checkpoint; use --resume")
        rng = np.random.default_rng(args.seed)
        # Spread sampling across the training chronology, never held-out years.
        selections = windows("train")[:args.chunks]
        ids = np.linspace(0, len(selections) - 1, args.normalization_chunks, dtype=int)
        total, squares, count = np.zeros(13), np.zeros(13), 0
        event("normalization_start", split="train", chunks=len(ids), windows=[selections[i] for i in ids])
        for chunk_number, ds in enumerate(cache.iterate([selections[i] for i in ids], "train"), 1):
            for i in range(0, ds.sizes["time"], 12):
                state, context = experiment.snapshot(ds, i)
                x = np.asarray(experiment.features(state, context)).reshape(-1, 13)
                total += x.sum(0); squares += (x*x).sum(0); count += len(x)
                event("normalization_progress", chunk=chunk_number, total_chunks=len(ids), snapshot=i, feature_rows=count)
        if not count:
            raise ValueError("no valid training snapshots for normalization")
        mean = total / count
        scale = np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 1e-6)
        atomic_bytes(args.output / "normalization.json", json.dumps({"mean": mean.tolist(), "scale": scale.tolist(),
                     "split": "train", "windows": [selections[i] for i in ids], "contract": contract}).encode())
        event("normalization_complete", feature_rows=count, path=str(args.output / "normalization.json"))
        params = initialize(jax.random.key(args.seed), constant=args.constant)
        first = second = jax.tree_util.tree_map(jnp.zeros_like, params)
        completed, position = 0, 0
    loss_fn = lambda params, batch: experiment.loss(params, batch, mean, scale, args.regularization)
    evaluate = jax.jit(lambda params, batch: experiment.loss(params, batch, mean, scale, 0.))
    baseline = jax.jit(lambda batch: experiment.loss(initialize(jax.random.key(0), constant=True), batch, mean, scale, 0.))
    if args.evaluate:
        scores = []
        event("evaluation_start", split=args.evaluate)
        for ds in cache.iterate(windows(args.evaluate)[:args.chunks], args.evaluate):
            for example in examples(experiment, ds, warmup):
                with phase("evaluation_example", example=len(scores) + 1):
                    scores.append([float(baseline(example)), float(evaluate(params, example))])
                event("evaluation_progress", examples=len(scores), physical_loss=scores[-1][0], neural_loss=scores[-1][1])
        values = np.asarray(scores)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("empty/nonfinite evaluation")
        report = {"split": args.evaluate, "examples": len(values), "physical_loss": float(values[:, 0].mean()),
                  "neural_loss": float(values[:, 1].mean()), "contract": contract}
        atomic_bytes(args.output / f"{args.evaluate}.json", json.dumps(report, indent=2).encode())
        event("evaluation_complete", **report)
        return
    selections = windows("train")[:args.chunks]
    steps_per_epoch = sum(max(0, stop - start - args.spinup - args.horizon)
                          for start, stop in selections) // len(devices)
    if not steps_per_epoch:
        raise ValueError("training selection cannot fill one global batch")
    event("training_plan", steps_per_epoch=steps_per_epoch, target_steps=args.steps, completed_steps=completed, dt_seconds=experiment.dt, horizon=args.horizon, validate_every_epochs=args.validate_every_epochs, validate_every_steps=args.validate_every_steps)
    with phase("validation_prepare", examples=args.validation_examples):
        subset, validation_windows = validation_subset(cache, experiment, warmup, args.validation_examples)
    physical_scores = []
    for index, example in enumerate(subset, 1):
        with phase("physical_validation", example=index, total=len(subset)):
            physical_scores.append(float(baseline(example)))
        event("physical_validation_progress", example=index, loss=physical_scores[-1])

    def check_validation(current_params, step, state=None):
        event("validation_start", step=step, examples=len(subset))
        report = validation_report(current_params, subset, evaluate, physical_scores, step, steps_per_epoch)
        report.update(windows=validation_windows, revision=args.revision,
                      spinup_snapshots=args.spinup, horizon=args.horizon)
        if step == 0 and not np.isclose(report["physical_loss"], report["neural_loss"], rtol=1e-9, atol=1e-9):
            raise ValueError("identity closure does not reproduce physical validation")
        atomic_bytes(args.output / "small_validation.json", json.dumps(report, indent=2).encode())
        with open(args.output / "validation.jsonl", "a") as stream:
            stream.write(json.dumps(report) + "\n")
        best_path = args.output / "best_validation.pkl"
        best = pickle.loads(best_path.read_bytes()) if best_path.exists() else None
        # Scores from different monitoring subsets are not directly comparable.
        comparable = best is not None and all(best["validation"][key] == report[key]
                                              for key in ("windows", "examples", "revision", "spinup_snapshots", "horizon"))
        if state is not None and (not comparable or report["neural_loss"] < best["validation"]["neural_loss"]):
            atomic_bytes(best_path, pickle.dumps(dict(state, validation=report)))
            event("best_checkpoint_saved", step=step, path=str(best_path), neural_loss=report["neural_loss"])
        event("small_validation", **report)

    if args.resume:
        order = saved["order"]
    else:
        order = rng.permutation(len(selections)).tolist()
        atomic_bytes(args.output / "order.json", json.dumps(order).encode())
    initial_checkpoint = dict(contract=contract, params=jax.device_get(params),
                              first=jax.device_get(first), second=jax.device_get(second),
                              mean=mean, scale=scale, completed=completed, position=position,
                              order=order, rng=rng.bit_generator.state)
    check_validation(params, completed, None if args.preflight_only else initial_checkpoint)
    if args.preflight_only:
        return
    replicate = lambda tree: device_batch([tree] * len(devices), devices)
    params, first, second = map(replicate, (params, first, second))
    count = replicate(jnp.asarray(completed))
    update = synchronized_update(loss_fn, args.learning_rate, devices)
    prepare = parallel_prepare(experiment, devices)
    event("execution_layout", training="one trajectory per device with averaged gradients",
          physical_warmup="parallel across selected devices",
          normalization_and_validation="single device", devices=[str(d) for d in devices])
    selections = windows("train")[:args.chunks]
    if sum(max(0, stop - start - args.spinup - args.horizon)
           for start, stop in selections) < len(devices):
        raise ValueError("training selection cannot fill one global batch")
    pending, seen = [], 0
    checked = False
    session_started = time.monotonic()
    session_start_step = completed
    while completed < args.steps:
        epoch_start_step = completed
        event("epoch_start", epoch=completed // steps_per_epoch + 1, resumed_position=position)
        for (start, stop), ds in cache.iterate([selections[i] for i in order], "train", indexed=True):
            available = max(0, stop - start - args.spinup - args.horizon)
            skip = min(available, max(0, position - seen))
            seen += skip
            if ds is None:
                seen += available - skip
                continue
            for example in examples(experiment, ds, warmup, skip, defer_warmup=True):
                seen += 1
                pending.append(example)
                if len(pending) != len(devices):
                    continue
                batch = device_batch(pending, devices)
                with phase("parallel_warmup", step=completed + 1, devices=len(devices),
                           spinup_intervals=args.spinup):
                    batch = jax.block_until_ready(prepare(batch))
                if not checked:
                    verify_device_placement(batch, devices, "physical_warmup")
                    first_example = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), batch)
                    with phase("physical_preflight"):
                        score = float(baseline(first_example))
                    if not np.isfinite(score):
                        raise ValueError("physical initialization rollout is nonfinite; optimization refused")
                    atomic_bytes(args.output / "physical_preflight.json", json.dumps({"loss": score,
                                 "dt_seconds": experiment.dt, "spinup_snapshots": args.spinup,
                                 "soil": "isothermal at initial observed surface temperature",
                                 "terrain": "flat", "dust": "initial snapshot held fixed; longwave=visible/3"}).encode())
                    event("physical_preflight_passed", loss=score)
                    checked = True
                update_started = time.monotonic()
                with phase("training_update", step=completed + 1, devices=len(devices), first_update_this_process=completed == session_start_step, includes_jit_compilation=completed == session_start_step):
                    result = jax.block_until_ready(update(params, first, second, count, batch))
                if completed == session_start_step:
                    verify_device_placement(result, devices, "training_update")
                params, first, second, count, losses, norms = result
                host = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), result)
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(host)):
                    raise ValueError("nonfinite loss/gradient/update; checkpoint not advanced")
                completed += 1
                position = seen
                pending.clear()
                saved = dict(contract=contract, params=host[0], first=host[1], second=host[2], mean=mean, scale=scale,
                             completed=completed, position=position, order=order, rng=rng.bit_generator.state)
                atomic_bytes(checkpoint, pickle.dumps(saved))
                event("checkpoint_saved", step=completed, path=str(checkpoint), bytes=checkpoint.stat().st_size)
                elapsed = time.monotonic() - session_started
                seconds_per_step = elapsed / (completed - session_start_step)
                record = dict(step=completed, total_steps=args.steps, epoch=completed / steps_per_epoch, loss=float(host[4]), gradient_norm=float(host[5]), position=position, learning_rate=args.learning_rate, update_seconds=time.monotonic() - update_started, elapsed_seconds=elapsed, eta_seconds=(args.steps-completed)*seconds_per_step, eta_scope="session average including compilation, data preparation and completed validation")
                with open(args.output / "training.jsonl", "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                event("training_progress", **record)
                if validation_due(completed, steps_per_epoch, args.validate_every_epochs,
                                  args.validate_every_steps, args.steps):
                    check_validation(host[0], completed, saved)
                if completed >= args.steps:
                    return
        # Drop the incomplete final batch consistently; next epoch is shuffled.
        event("epoch_complete", completed_steps=completed, dropped_examples=len(pending))
        if completed == epoch_start_step and position == 0:
            raise ValueError("valid training windows cannot fill one device batch")
        pending.clear()
        position = seen = 0
        order = rng.permutation(len(selections)).tolist()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        LOGGER.exception("run_failed")
        raise
    else:
        event("run_complete")
