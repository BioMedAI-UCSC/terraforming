#!/usr/bin/env python3
"""Coupled gradient, calibration, neural-capability and paired-ablation experiments.

Targets are explicit matched trajectories, never silently rounded or interpolated.
See docs/differentiable-experiments.md for the manifest contract and commands.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/mars-calibration"))
from mars_calibration import paper as p
from mars_calibration import driver as d
from src.framework.neural import NeuralTendency, state_features, fit_normalization
from src.framework.gcm.restart import save_restart


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_manifest(path):
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not data.get("windows"):
        raise ValueError("Expected schema_version=1 and nonempty windows")
    ids, hashes = set(), set()
    intervals = []
    for row in data["windows"]:
        if row["id"] in ids or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unique IDs and explicit train/validation/test splits required")
        ids.add(row["id"])
        for key in ("restart", "target", *(("forcing",) if "forcing" in row else ())):
            row[key] = str((path.parent / row[key]).resolve())
            if digest(row[key]) != row[key + "_sha256"]:
                raise ValueError(f"Hash mismatch: {row[key]}")
        if row["restart_sha256"] in hashes:
            raise ValueError("Repeated restart is not an independent window")
        hashes.add(row["restart_sha256"])
        if row["reference_kind"] not in ("synthetic", "reanalysis", "observation"):
            raise ValueError("Explicit reference_kind required")
        if not row.get("source") or not row.get("block"):
            raise ValueError("Source provenance and temporal block are required")
        lo, hi = row["start_seconds"], row["end_seconds"]
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("Invalid absolute time interval")
        for source, start, end in intervals:
            if source == row["source"] and max(start, lo) < min(end, hi):
                raise ValueError("Overlapping windows are forbidden")
        intervals.append((row["source"], lo, hi))
    blocks = {}
    for row in data["windows"]:
        key = (row["source"], row["block"])
        if key in blocks and blocks[key] != row["split"]:
            raise ValueError("Temporal blocks must not cross splits")
        blocks[key] = row["split"]
    return data


def target_arrays(row, model, dt):
    if "forcing" in row:
        with np.load(row["forcing"], allow_pickle=False) as f:
            model.forcing = dataclasses.replace(model.base_forcing,
                init_orbital_angle_rad=float(f["init_orbital_angle_rad"]),
                dust_visible_optical_depth=d.jnp.asarray(f["dust_visible"]),
                dust_longwave_optical_depth=d.jnp.asarray(f["dust_longwave"]),
                dust_climatology_ls_deg=None, dust_visible_climatology=None,
                dust_longwave_climatology=None)
    else:
        model.forcing = model.base_forcing
    with np.load(row["target"], allow_pickle=False) as z:
        seconds, fields = z["seconds"].copy(), z["fields"].copy()
    expected = (len(seconds), 3, 12, 64, 32)
    if fields.shape != expected or not np.isfinite(fields).all():
        raise ValueError(f"Expected finite T(K),u(m/s),v(m/s) fields {expected}")
    if (not np.isfinite(seconds).all() or len(seconds) == 0 or seconds[0] <= 0
            or np.any(np.diff(seconds) <= 0)
            or not np.allclose(seconds / dt, np.round(seconds / dt), rtol=0, atol=1e-9)
            or seconds[-1] > row["end_seconds"] - row["start_seconds"] + 1e-6):
        raise ValueError("Target times must be positive increasing exact timestep multiples within window")
    model.initial = d.load_restart(Path(row["restart"]))
    model.diagnostics(model.initial)
    return seconds, d.jnp.asarray(fields)


def neural_rollout(model, seconds, dt, adapter, norm):
    initial = model.initial
    forcing = model.forcing
    counts = np.diff(np.r_[0, np.rint(seconds / dt)]).astype(int)
    def run(params):
        component = adapter.bind(params, norm, coords=model.coords, specs=model.specs,
                                 body=d.MARS_BODY_3D, forcing=forcing)
        step = d._build_step(model.coords, model.specs, forcing, dt,
                             orography=model.orography, neural_tendency=component)
        state, outputs = initial, []
        # Static scan lengths support reverse mode; rematerialization bounds step storage.
        for count in counts:
            state = d.jax.lax.scan(lambda s, _: (step(s), None), state, None, length=int(count))[0]
            outputs.append(model.observe(state))
        return d.jnp.stack(outputs)
    return run


def errors(predicted, target, model):
    mse = np.mean(np.sum((np.asarray(predicted) - np.asarray(target)) ** 2
                         * np.asarray(model.weights), axis=(-2, -1)), axis=-1)
    return np.sqrt(mse)


def check_directions(objective, x, epsilons, directions, reverse=False):
    f, cost = p.compile_function(objective, x)
    derivative = d.jax.grad(objective) if reverse else d.jax.jacfwd(objective)
    gfun, gcost = p.compile_function(derivative, x)
    started = time.perf_counter()
    gradient = np.asarray(p.block(gfun(x)))
    elapsed = time.perf_counter() - started
    rows = []
    for i, direction in enumerate(directions):
        ad = float(gradient @ direction)
        for epsilon in epsilons:
            fd = float((f(x + epsilon * direction) - f(x - epsilon * direction)) / (2 * epsilon))
            absolute = abs(ad - fd)
            rows.append(dict(direction=i, epsilon=epsilon, autodiff=ad, finite_difference=fd,
                absolute_error=absolute, relative_error=absolute / max(abs(ad), abs(fd), 1e-12),
                pass_check=bool(np.isfinite([ad, fd]).all() and absolute <= 1e-8 + 1e-4 * abs(fd)),
                gradient_seconds=elapsed, forward_compile_seconds=cost["compile_seconds"],
                gradient_compile_seconds=gcost["compile_seconds"]))
    return rows


def gradients(model, args):
    rows = []
    parameters = d.jnp.asarray([0.6, 0.8, 1.4])
    for steps in args.steps:
        run = model.trajectory([steps * args.dt], args.dt)
        # A fixed nonzero-temperature objective avoids a zero gradient at a synthetic optimum.
        def objective(x):
            fields = run(x)
            return d.jnp.sum(fields[0, 0] * model.weights) / 12
        checks = check_directions(objective, parameters, args.epsilons, np.eye(3))
        rows.extend(dict(steps=steps, seconds=steps * args.dt, **r) for r in checks)
        p.write_csv(args.output / "gradients.csv", rows)
        print(f"gradient horizon {steps} steps complete", flush=True)
        d.jax.clear_caches()
    p.dump(args.output / "acceptance.json", {"all_checks_pass": all(r["pass_check"] for r in rows),
        "scope": "Physical forward-mode temperature-objective derivatives at the supplied restart"})


def prepare(model, args):
    """Distinct restarts on one synthetic trajectory; explicitly not independent climates."""
    windows = []
    teacher = d.jnp.asarray([0.6, 0.8, 1.4])
    steps = args.steps[-1]
    for i, split in enumerate(("train", "train", "validation", "test")):
        start = i * 2 * steps * args.dt
        restart = args.output / f"restart-{i}.npz"
        save_restart(model.initial, restart)
        times = np.asarray([steps * args.dt])
        run = model.trajectory(times, args.dt)
        fields = np.asarray(p.block(d.jax.jit(run)(teacher)))
        target = args.output / f"target-{i}.npz"
        np.savez_compressed(target, seconds=times, fields=fields)
        windows.append(dict(id=f"synthetic-{i}", split=split, block=str(i),
            source="same-model synthetic continuation", reference_kind="synthetic",
            start_seconds=start, end_seconds=start + times[-1], restart=restart.name,
            target=target.name, restart_sha256=digest(restart), target_sha256=digest(target)))
        advance = model.trajectory([2 * steps * args.dt], args.dt, states=True)
        history = p.block(d.jax.jit(advance)(teacher))
        model.initial = d.jax.tree_util.tree_map(lambda x: x[-1], history)
    p.dump(args.output / "windows.json", dict(schema_version=1, windows=windows))


def fit(model, args, manifest, neural):
    windows = manifest["windows"]
    if {w["split"] for w in windows} != {"train", "validation", "test"}:
        raise ValueError("Fitting requires train, validation and test windows")
    loaded, features = [], []
    for row in windows:
        seconds, target = target_arrays(row, model, args.dt)
        loaded.append((row, seconds, target, model.initial, model.forcing))
        if row["split"] == "train":
            features.append(np.asarray(state_features(model.initial, model.coords, model.specs,
                                                       d.MARS_BODY_3D, model.forcing)))
    train_targets = d.jnp.concatenate([t for r, _, t, _, _ in loaded if r["split"] == "train"])
    scale = d.jnp.maximum(d.jnp.std(train_targets, axis=(0, 2, 3, 4)), 1.)
    norm = fit_normalization(np.stack(features), split="train")
    adapter = NeuralTendency(12, maximum_heating_k_s=args.heating_bound, hidden_sizes=(8,))
    params = adapter.init(d.jax.random.PRNGKey(args.seed))
    params = params[:-1] + ((d.jnp.zeros_like(params[-1][0]), d.jnp.zeros_like(params[-1][1])),)
    from jax.flatten_util import ravel_pytree
    neural_x, unravel = ravel_pytree(params)
    x0 = neural_x if neural else d.jnp.asarray((np.array([1., 1., 1.]) - d.LOWER_BOUNDS) / (d.UPPER_BOUNDS - d.LOWER_BOUNDS))
    runs = []
    for row, seconds, target, initial, forcing in loaded:
        model.initial = initial
        model.forcing = forcing
        if neural:
            raw = neural_rollout(model, seconds, args.dt, adapter, norm)
            run = lambda x, raw=raw: raw(unravel(x))
        else:
            raw = model.trajectory(seconds, args.dt)
            run = lambda x, raw=raw: raw(d.jnp.asarray(d.LOWER_BOUNDS) + x * d.jnp.asarray(d.UPPER_BOUNDS - d.LOWER_BOUNDS))
        # Capture each physical restart when constructing a trajectory callable.
        physical = model.trajectory(seconds, args.dt)
        baseline = np.asarray(p.block(d.jax.jit(physical)(d.jnp.ones(3))))
        if neural:
            identity = np.asarray(p.block(d.jax.jit(run)(x0)))
            if not np.allclose(identity, baseline, atol=1e-9, rtol=1e-9):
                raise ValueError("Zero-output neural baseline identity failed")
        runs.append((row, seconds, target, run, baseline))
    def loss(x, split):
        return sum(model.loss(run(x), t, scale) for r, _, t, run, _ in runs if r["split"] == split) / sum(r["split"] == split for r, *_ in runs)
    objective = lambda x: loss(x, "train")
    if neural:
        direction = np.random.default_rng(args.seed).normal(size=x0.shape)
        direction /= np.linalg.norm(direction)
        checks = check_directions(objective, x0, args.epsilons, [direction], reverse=True)
    else:
        checks = check_directions(objective, x0, args.epsilons, np.eye(3))
    p.write_csv(args.output / "gradient-checks.csv", checks)
    if not all(r["pass_check"] for r in checks):
        raise ValueError("Gradient checks failed; inspect artifact before training")
    vg = d.jax.value_and_grad(objective) if neural else d.jax.jacfwd(lambda x: (objective(x), objective(x)), has_aux=True)
    if not neural:
        tangent = vg
        vg = lambda x: tuple(reversed(tangent(x)))
    vg, _ = p.compile_function(vg, x0)
    validation, _ = p.compile_function(lambda x: loss(x, "validation"), x0)
    best, best_loss = x0, float(validation(x0))
    x, m, v, trace = x0, d.jnp.zeros_like(x0), d.jnp.zeros_like(x0), []
    for update in range(1, args.updates + 1):
        started = time.perf_counter()
        value, grad = p.block(vg(x))
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite training loss/gradient")
        grad = grad / d.jnp.maximum(1., d.jnp.linalg.norm(grad))
        m, v = .9*m + .1*grad, .999*v + .001*grad**2
        x = x - args.learning_rate * (m/(1-.9**update)) / (d.jnp.sqrt(v/(1-.999**update)) + 1e-8)
        if not neural:
            x = d.jnp.clip(x, 0., 1.)
        val = float(validation(x))
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        if val < best_loss:
            best, best_loss = x, val
        trace.append(dict(update=update, train_loss_before_update=float(value), validation_loss=val,
                          seconds=time.perf_counter()-started))
        p.write_csv(args.output / "training.csv", trace)
        np.savez(args.output / "checkpoint.npz", current=np.asarray(x), best=np.asarray(best),
                 normalization_mean=np.asarray(norm.mean), normalization_scale=np.asarray(norm.scale),
                 field_scale=np.asarray(scale), update=update)
        print(json.dumps(trace[-1]), flush=True)
    rows = []
    for row, seconds, target, run, baseline in runs:
        predicted = np.asarray(p.block(d.jax.jit(run)(best)))
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite evaluation trajectory")
        for name, values in (("physical", baseline), ("neural" if neural else "calibrated", predicted)):
            rmse = errors(values, target, model)
            for i, seconds_i in enumerate(seconds):
                rows.append(dict(window=row["id"], split=row["split"], block=row["block"],
                    reference_kind=row["reference_kind"], model=name, seconds=float(seconds_i),
                    temperature_rmse_k=float(rmse[i, 0]), u_rmse_ms=float(rmse[i, 1]), v_rmse_ms=float(rmse[i, 2])))
    p.write_csv(args.output / "skill.csv", rows)
    p.dump(args.output / "acceptance.json", dict(gradient_checks_pass=True,
        identity_checked=neural, validation_loss=best_loss,
        scope="Capability experiment; held-out skill is reported regardless of improvement. Synthetic continuation is not independent climate validation."))


def ablations(model, args, manifest):
    rows = []
    for row in manifest["windows"]:
        seconds, _ = target_arrays(row, model, args.dt)
        baseline = None
        for case in p.ablation_cases([1., 1., 1.], args.dt):
            if case["name"] not in args.cases and case["name"] != "full":
                continue
            kw = {k: case[k] for k in ("overrides", "co2_exchange", "diffusion_sols") if k in case}
            run = model.trajectory(seconds, case["dt"], **kw)
            prediction = np.asarray(p.block(d.jax.jit(run)(d.jnp.asarray(case["parameters"]))))
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite ablation {row['id']} {case['name']}")
            if baseline is None:
                baseline = prediction
            rms = errors(prediction, baseline, model)
            rows.append(dict(window=row["id"], block=row["block"], case=case["name"],
                temperature_rmse_k=float(np.sqrt(np.mean(rms[:, 0]**2))),
                u_rmse_ms=float(np.sqrt(np.mean(rms[:, 1]**2))), v_rmse_ms=float(np.sqrt(np.mean(rms[:, 2]**2)))))
            p.write_csv(args.output / "ablations.csv", rows)
        d.jax.clear_caches()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["prepare-synthetic", "gradients", "calibration", "neural", "ablations"])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--dt", type=float, help="Default: manifest dt_seconds, otherwise 300")
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 4, 16, 32, 74])
    ap.add_argument("--epsilons", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5])
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=.001)
    ap.add_argument("--heating-bound", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--cases", nargs="+", default=["no_pbl", "no_convection", "no_regolith", "half_timestep"])
    args = ap.parse_args()
    manifest = load_manifest(args.manifest) if args.manifest else None
    if args.dt is None:
        args.dt = manifest.get("dt_seconds", 300.) if manifest else 300.
    if (not 0 < args.dt <= 300 or min(args.steps) < 1 or args.steps != sorted(set(args.steps))
            or args.updates < 1 or not np.isfinite(args.epsilons).all() or min(args.epsilons) <= 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        ap.error("Invalid timestep, ordered unique horizons, updates, epsilons or learning rate")
    known = {c["name"] for c in p.ablation_cases([1., 1., 1.], args.dt)}
    if set(args.cases) - known:
        ap.error("Unknown ablation case")
    inputs = p.validate_inputs(args.inputs)
    if manifest and "native_contract" in manifest:
        if digest(args.inputs / "mola.img") != manifest["native_contract"]["terrain_sha256"]:
            raise ValueError("Native initialization and simulator terrain must match")
    if args.task in ("calibration", "neural", "ablations") and manifest is None:
        ap.error("This task requires --manifest")
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    d.jax.config.update("jax_enable_x64", True)
    if args.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    model = p.Model(args.inputs)
    model.base_forcing = model.forcing
    if manifest:
        for row in manifest["windows"]:
            target_arrays(row, model, args.dt)
        model.initial = d.load_restart(args.inputs / "restart.npz")
    if args.validate_only:
        print("Validated inputs and requested manifest; no trajectories executed")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    p.dump(args.output / "provenance.json", dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        inputs=inputs, manifest=manifest, environment=p.environment(),
        source_hashes={str(f.relative_to(ROOT)): digest(f) for folder in (ROOT/'scripts', ROOT/'package/src', ROOT/'apps/mars-calibration') for f in folder.rglob('*.py')}))
    try:
        if args.task == "prepare-synthetic": prepare(model, args)
        elif args.task == "gradients": gradients(model, args)
        elif args.task == "ablations": ablations(model, args, manifest)
        else: fit(model, args, manifest, args.task == "neural")
    except Exception as exc:
        p.dump(args.output / "failure.json", dict(error=str(exc)))
        raise
    p.dump(args.output / "complete.json", dict(task=args.task, status="complete"))


if __name__ == "__main__":
    main()
