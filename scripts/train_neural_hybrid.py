#!/usr/bin/env python3
"""Train a NeuralGCM-inspired Mars hybrid on one host with synchronized GPUs."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from itertools import islice
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from neural_logging import LOGGER, configure, event, phase
from neural_macda import NativeCache, atomic_bytes, locked, windows
from neural_hybrid import HybridExperiment, examples, fit_statistics, BODY
from train_neural_pbl import device_batch, synchronized_update, verify_device_placement
from src.framework.gcm._dinosaur import jax, jnp
from src.framework.physics.neural_column import initialize


def curriculum(value):
    try:
        stages = [tuple(map(int, item.split(":"))) for item in value.split(",")]
        if any(len(x) != 2 or min(x) < 1 for x in stages):
            raise ValueError()
        if any(a[0] > b[0] or a[1] >= b[1] for a, b in zip(stages, stages[1:])):
            raise ValueError()
        return stages
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("use nondecreasing horizons and increasing end steps, e.g. 3:1000,6:2000,12:4000") from None


def stage_horizon(stages, completed):
    for horizon, end in stages:
        if completed < end:
            return horizon
    return stages[-1][0]


def active_params(params, ablation):
    disabled = {"tendency-only": ("encoder", "decoder"), "adapters-only": ("physics",),
                "decoder-only": ("encoder", "physics")}.get(ablation, ())
    return {name: (jax.tree_util.tree_map(jnp.zeros_like, network) if name in disabled else network)
            for name, network in params.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True)
    p.add_argument("--cache", default="data/neural_macda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mola", type=Path)
    p.add_argument("--flat-terrain", action="store_true")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--curriculum", type=curriculum, default=curriculum("3:1000,6:2000,12:4000"))
    p.add_argument("--validation-horizon", type=int, default=3)
    p.add_argument("--validation-examples", type=int, default=8)
    p.add_argument("--validate-every-steps", type=int, default=100)
    p.add_argument("--validate-every-epochs", type=int, default=1)
    p.add_argument("--normalization-chunks", type=int, default=7)
    p.add_argument("--chunks", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--max-dt", type=float, default=300.)
    p.add_argument("--refresh-seconds", type=float, default=1800.)
    p.add_argument("--spinup", type=int, default=12, help="past surface snapshots for soil initialization; atmosphere is not spun up")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--regularization", type=float, default=1e-4)
    p.add_argument("--ablation", choices=("full", "tendency-only", "adapters-only", "decoder-only"), default="full")
    p.add_argument("--export-validation-predictions", action="store_true",
                   help="save paired validation forecasts for a separate equal-weight ensemble")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--evaluate", choices=("validation", "test", "storm"))
    p.add_argument("--stage", choices=("train", "validation", "test", "storm"))
    args = p.parse_args()
    positive = (args.devices, args.steps, args.width, args.blocks, args.normalization_chunks,
                args.validation_examples, args.validation_horizon)
    if min(positive) < 1 or args.spinup < 0 or (args.chunks is not None and args.chunks < 1):
        p.error("counts must be positive, spinup nonnegative")
    if min(args.validate_every_steps, args.validate_every_epochs) < 0:
        p.error("validation intervals must be nonnegative")
    if any(not np.isfinite(x) or x <= 0 for x in (args.max_dt, args.refresh_seconds, args.learning_rate)) or not np.isfinite(args.regularization) or args.regularization < 0:
        p.error("invalid timestep, refresh interval, learning rate, or regularization")
    if args.steps > args.curriculum[-1][1]:
        p.error("curriculum must cover requested training steps")
    if args.spinup + max(args.validation_horizon, args.curriculum[-1][0]) >= 120:
        p.error("soil history plus horizon must fit within a 120-snapshot chunk")
    if sum(bool(x) for x in (args.stage, args.evaluate, args.preflight_only)) > 1:
        p.error("stage, evaluate, and preflight-only are mutually exclusive")
    if not args.stage and (bool(args.mola) == args.flat_terrain):
        p.error("supply --mola PATH or explicitly select --flat-terrain")
    configure(args.output)
    jax.config.update("jax_enable_x64", True)
    event("run_start", arguments=vars(args), python=sys.version, jax_version=jax.__version__, architecture="neuralgcm-inspired-column-v1")
    cache = NativeCache(args.cache, args.revision)
    if args.stage:
        for ds in cache.iterate(windows(args.stage)[:args.chunks], args.stage):
            event("staging_progress", first_sol=float(ds.time[0]), last_sol=float(ds.time[-1]))
        return
    devices = jax.local_devices()[:args.devices]
    if len(devices) != args.devices or (not args.cpu and any(d.platform != "gpu" for d in devices)):
        p.error(f"requested {args.devices} GPUs; available: {jax.local_devices()}")
    event("devices_selected", devices=[str(d) for d in devices], global_batch_size=len(devices),
          parallel_stages=["training_update"], single_device_stages=["normalization", "validation", "snapshot_and_soil_preparation"])
    with locked(args.output / "run.lock"):
        run(args, cache, devices)


def run(args, cache, devices):
    e = HybridExperiment(args.max_dt, spinup=args.spinup, mola=args.mola,
                         flat_terrain=args.flat_terrain, refresh_seconds=args.refresh_seconds)
    contract = {key: getattr(args, key) for key in ("revision", "seed", "width", "blocks", "max_dt", "refresh_seconds",
                "spinup", "learning_rate", "regularization", "normalization_chunks", "chunks", "devices", "curriculum", "ablation")}
    contract.update(schema="neuralgcm-inspired-column-v1", terrain_sha256=hashlib.sha256(e.terrain.tobytes()).hexdigest(),
                    validation_horizon=args.validation_horizon, validation_examples=args.validation_examples)
    checkpoint = args.output / "checkpoint.pkl"
    selections = windows("train")[:args.chunks]
    reserved_horizon = args.curriculum[-1][0]
    available = [stop-start-args.spinup-reserved_horizon for start, stop in selections]
    steps_per_epoch = sum(available) // len(devices)
    if steps_per_epoch < 1:
        raise ValueError("not enough training examples for one device batch")
    rng = np.random.default_rng(args.seed)
    if args.resume or args.evaluate:
        event("checkpoint_loading", path=str(checkpoint))
        saved = pickle.loads(checkpoint.read_bytes())  # Trusted, locally produced files only.
        if saved["contract"] != contract:
            raise ValueError("checkpoint architecture/data/training contract differs; use a separate output directory")
        params, first, second = (saved[k] for k in ("params", "first", "second"))
        stats, completed, position, order = (saved[k] for k in ("stats", "completed", "position", "order"))
        rng.bit_generator.state = saved["rng"]
        event("checkpoint_loaded", step=completed, position=position)
    else:
        if checkpoint.exists():
            raise ValueError("checkpoint exists; use --resume or a new output")
        ids = np.unique(np.linspace(0, len(selections)-1, min(args.normalization_chunks, len(selections)), dtype=int))
        chosen = [selections[i] for i in ids]
        def training_chunks():
            for i, ds in enumerate(cache.iterate(chosen, "train"), 1):
                event("normalization_progress", chunk=i, chunks=len(chosen), split="train")
                yield ds
        with phase("normalization", windows=chosen, split="train"):
            stats = fit_statistics(e, training_chunks())
        atomic_bytes(args.output / "normalization.json", json.dumps(dict(stats={k:v.tolist() for k,v in stats.items()}, windows=chosen, contract=contract)).encode())
        params = initialize(jax.random.key(args.seed), len(stats["mean"]), e.coords.vertical.layers, width=args.width, blocks=args.blocks)
        first = second = jax.tree_util.tree_map(jnp.zeros_like, params)
        completed = position = 0
        order = rng.permutation(len(selections)).tolist()
    event("training_plan", steps_per_epoch=steps_per_epoch, target_steps=args.steps, curriculum=args.curriculum,
          dt_seconds=e.dt, refresh_steps=e.refresh_steps, parameters=sum(x.size for x in jax.tree_util.tree_leaves(params)),
          initialization="observed atmosphere; past-only implicit soil conduction", terrain_sha256=contract["terrain_sha256"])
    horizon = args.validation_horizon
    export_predictions = getattr(args, "export_validation_predictions", False)
    metric_options = {"return_forecasts": True} if export_predictions else {}
    evaluate = jax.jit(lambda p, x: e.metrics(active_params(p, args.ablation), x, stats, horizon, **metric_options))
    physical = jax.jit(lambda x: e.metrics(None, x, stats, horizon, physical=True)[0])
    def subset():
        choices = windows("validation")
        chosen = [choices[i] for i in np.linspace(0, len(choices)-1, min(args.validation_examples, 4), dtype=int)]
        result = []
        for i, ds in enumerate(cache.iterate(chosen, "validation")):
            count = math_ceil_div(args.validation_examples-len(result), len(chosen)-i)
            result.extend(islice(examples(e, ds, horizon), count))
        if len(result) != args.validation_examples:
            raise ValueError("not enough held-out validation examples")
        return result, chosen
    if args.evaluate:
        results = []
        for ds in cache.iterate(windows(args.evaluate)[:args.chunks], args.evaluate):
            for x in examples(e, ds, horizon):
                with phase("evaluation_example", split=args.evaluate, example=len(results)+1):
                    results.append([float(physical(x)), float(evaluate(params, x)[0])])
                event("evaluation_progress", physical_loss=results[-1][0], neural_loss=results[-1][1], examples=len(results))
        values = np.asarray(results)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("empty or nonfinite evaluation")
        report = dict(split=args.evaluate, horizon=horizon, examples=len(values), physical_loss=float(values[:,0].mean()), neural_loss=float(values[:,1].mean()), contract=contract)
        atomic_bytes(args.output / f"{args.evaluate}.json", json.dumps(report, indent=2).encode())
        event("evaluation_complete", **report)
        return
    with phase("validation_prepare", examples=args.validation_examples):
        holdout, holdout_windows = subset()
    baseline = []
    for i, x in enumerate(holdout):
        with phase("physical_validation", example=i+1):
            baseline.append(float(physical(x)))
    if not np.isfinite(baseline).all():
        raise ValueError("physical initialization rollout is nonfinite; optimization refused")

    def state_dict(p, m, v):
        return dict(contract=contract, params=jax.device_get(p), first=jax.device_get(m), second=jax.device_get(v),
                    stats=stats, completed=completed, position=position, order=order, rng=rng.bit_generator.state)

    def validate(p, saved):
        scores, latent_scores, layer_rmse = [], [], []
        predictions = []
        for i, x in enumerate(holdout):
            with phase("validation_example", step=completed, example=i+1):
                metrics = evaluate(p, x)
                scores.append(float(metrics[0]))
                latent_scores.append(float(metrics[2]))
                layer_rmse.append(np.asarray(metrics[3]))
                if export_predictions:
                    predictions.append(np.asarray(metrics[4]))
        if not all(np.isfinite(values).all() for values in (scores, latent_scores, layer_rmse)):
            raise ValueError("nonfinite validation; checkpoint retained for diagnosis")
        if completed == 0 and not np.allclose(scores, baseline, rtol=1e-9, atol=1e-9):
            raise ValueError("zero-initialized hybrid differs from physical baseline")
        active = active_params(p, args.ablation)
        state, context, _, _ = holdout[0]
        rates = np.asarray(e.output(active["physics"], state, context, stats, tendency=True))
        area = np.asarray(e.coords.horizontal.quadrature_weights)
        ps = np.asarray(jnp.exp(e.coords.horizontal.to_nodal(state.dynamics.log_surface_pressure)))[0] * e.pressure_scale
        dsigma = np.diff(e.coords.vertical.boundaries)[:, None, None]
        mass = ps * dsigma / BODY.gravity_m_s2
        report = dict(step=completed, epoch=completed/steps_per_epoch, split="validation", examples=len(scores), horizon=horizon,
                      neural_loss=float(np.mean(scores)), physical_loss=float(np.mean(baseline)), windows=holdout_windows,
                      paired_improvements=(np.asarray(baseline)-scores).tolist(),
                      latent_loss=float(np.mean(latent_scores)),
                      rmse_by_field_layer=np.sqrt(np.mean(np.asarray(layer_rmse)**2, axis=0)).tolist(),
                      tendency_rms_si=np.sqrt(np.mean(rates**2, axis=(1,2,3))).tolist(),
                      net_temperature_mass_source_K_kg_m2_s=float(np.sum(rates[0]*mass*area)/np.sum(area)))
        if export_predictions:
            # Same normalization, cases, targets and weights are required before
            # another process may average forecasts across methods.
            comparison = {key: value for key, value in contract.items() if key != "ablation"}
            comparison.update(windows=holdout_windows,
                              state_times=[float(x[0].dynamics.sim_time) for x in holdout],
                              stats_sha256=hashlib.sha256(json.dumps(
                                  {k: np.asarray(v).tolist() for k, v in stats.items()}, sort_keys=True).encode()).hexdigest())
            artifact = args.output / "validation-predictions" / f"step-{completed:06d}.npz"
            artifact.parent.mkdir(exist_ok=True)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, prediction=np.stack(predictions),
                                target=np.stack([x[2][:horizon] for x in holdout]),
                                mass=np.stack([x[3][:horizon] for x in holdout]),
                                field_scale=stats["field_scale"], physical_losses=baseline, latent_losses=latent_scores,
                                metadata=json.dumps(comparison, sort_keys=True), step=completed)
            atomic_bytes(artifact, buffer.getvalue())
            report["predictions_path"] = str(artifact)
        with open(args.output / "validation.jsonl", "a") as stream:
            stream.write(json.dumps(report)+"\n")
        best_path = args.output / "best_validation.pkl"
        best = pickle.loads(best_path.read_bytes())["validation"]["neural_loss"] if best_path.exists() else float("inf")
        if report["neural_loss"] < best:
            atomic_bytes(best_path, pickle.dumps(dict(saved, validation=report)))
            event("best_checkpoint_saved", step=completed, path=str(best_path))
        atomic_bytes(args.output / "small_validation.json", json.dumps(report, indent=2).encode())
        event("small_validation", **report)

    saved = state_dict(params, first, second)
    validate(params, saved)
    atomic_bytes(checkpoint, pickle.dumps(saved))
    if args.preflight_only:
        event("preflight_passed", checkpoint=str(checkpoint), next_action="resume with identical model/data options")
        return
    replicate = lambda tree: device_batch([tree]*len(devices), devices)
    params, first, second = map(replicate, (params, first, second))
    count = replicate(np.asarray(completed))
    updates = {}
    start_time, start_step = time.monotonic(), completed
    while completed < args.steps:
        epoch_start_step = completed
        pending, seen = [], 0
        for (start, stop), ds in cache.iterate([selections[i] for i in order], "train", indexed=True):
            size = max(0, stop - start - args.spinup - reserved_horizon)
            skip = min(size, max(0, position-seen)); seen += skip
            if ds is None:
                seen += size - skip
                continue
            for x in examples(e, ds, reserved_horizon, skip):
                pending.append(x); seen += 1
                if len(pending) != len(devices):
                    continue
                h = stage_horizon(args.curriculum, completed)
                if h not in updates:
                    event("curriculum_stage", horizon=h, step=completed, includes_jit_compilation=True)
                    updates[h] = synchronized_update(lambda p, x, h=h: e.loss(active_params(p, args.ablation), x, stats, h, args.regularization), args.learning_rate, devices)
                with phase("training_update", step=completed+1, horizon=h, devices=len(devices)):
                    result = jax.block_until_ready(updates[h](params, first, second, count, device_batch(pending, devices)))
                if completed == start_step:
                    verify_device_placement(result, devices, "training_update")
                host = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), result)
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(host)):
                    raise ValueError("nonfinite loss/gradient/parameters; checkpoint not advanced")
                params, first, second, count = result[:4]
                completed += 1; position = seen; pending.clear()
                saved = state_dict(*host[:3])
                atomic_bytes(checkpoint, pickle.dumps(saved))
                elapsed = time.monotonic()-start_time
                record = dict(step=completed, total_steps=args.steps, horizon=h, epoch=completed/steps_per_epoch,
                              loss=float(host[4]), gradient_norm=float(host[5]), elapsed_seconds=elapsed,
                              eta_seconds=elapsed/(completed-start_step)*(args.steps-completed))
                with open(args.output / "training.jsonl", "a") as stream:
                    stream.write(json.dumps(record)+"\n")
                event("training_progress", **record)
                event("checkpoint_saved", step=completed, path=str(checkpoint))
                if completed == args.steps or (args.validate_every_steps and completed % args.validate_every_steps == 0) or (args.validate_every_epochs and completed % (args.validate_every_epochs*steps_per_epoch) == 0):
                    validate(host[0], saved)
                if completed >= args.steps:
                    return
        event("epoch_complete", step=completed, dropped_examples=len(pending))
        if completed == epoch_start_step and position == 0:
            raise ValueError("valid training windows cannot fill one device batch")
        position = 0
        order = rng.permutation(len(selections)).tolist()


def math_ceil_div(a, b):
    return (a+b-1)//b


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        LOGGER.exception("run_failed")
        raise
    else:
        event("run_complete")
