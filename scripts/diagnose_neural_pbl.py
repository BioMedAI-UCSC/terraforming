#!/usr/bin/env python3
"""Read-only checkpoint diagnosis on the fixed MACDA validation subset."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from neural_logging import LOGGER, configure, event, phase
from neural_macda import NativeCache, atomic_bytes
from train_neural_pbl import Experiment, validation_subset
from src.framework.gcm._dinosaur import jax, jnp
from src.framework.physics.neural_pbl import Closure, FEATURE_NAMES, initialize, multipliers

FIELDS = ("temperature_k", "eastward_wind_ms", "northward_wind_ms")


def error_metrics(predicted, target, mass):
    """Same forecast loss as training, plus physical-unit per-layer RMSE."""
    squared = (np.asarray(predicted) - np.asarray(target)) ** 2
    mass = np.asarray(mass)
    numerator = np.sum(squared * mass[:, None], axis=(0, 3, 4))
    denominator = np.sum(mass, axis=(0, 2, 3))
    return {"loss": float(numerator.sum() / (300 * denominator.sum())),
            "layer_mse": numerator / denominator[None]}


def correction_summary(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("empty/nonfinite correction sample")
    # Within 1% of the permitted log-multiplier interval's endpoints.
    scaled = np.log(values) / np.log(2.)
    return {"minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "fraction_near_lower_bound": float(np.mean(scaled <= -.98)),
            "fraction_near_upper_bound": float(np.mean(scaled >= .98))}


def shifted(params, channel, delta):
    result = dict(params)
    key = "constant" if "constant" in params else "b3"
    value = np.array(params[key], copy=True)
    value[channel] += delta
    result[key] = value
    return result


def render_summary(report):
    aggregate = report["variants"]
    physical_loss = aggregate["physical"]["loss"]
    lines = ["# Neural PBL checkpoint diagnosis", "",
             f"Checkpoint step: {report['step']}; validation examples: {report['examples']}",
             f"Physical loss: {physical_loss:.10g}",
             f"Neural loss: {aggregate['neural']['loss']:.10g}",
             f"Loss improvement (%): {report['improvement_percent']}", "",
             "## Corrections below PBL height (initial interfaces)", "",
             "| Coefficient | Min | Mean | Max | Std | Near 0.5 (%) | Near 2 (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in report["corrections"]["below_pbl_height"].items():
        lines.append(f"| {name} | {stats['minimum']:.5g} | {stats['mean']:.5g} | {stats['maximum']:.5g} | {stats['std']:.5g} | {100*stats['fraction_near_lower_bound']:.2f} | {100*stats['fraction_near_upper_bound']:.2f} |")
    lines.extend(["", "## RMSE by layer: physical / neural", "",
                  "| Sigma (top to bottom) | Temperature (K) | East wind (m/s) | North wind (m/s) |",
                  "|---|---:|---:|---:|"])
    for k, sigma in enumerate(report["sigma_top_to_bottom"]):
        row = [f"{sigma:.5f}"]
        for field in FIELDS:
            row.append(f"{aggregate['physical']['layer_rmse'][field][k]:.6g} / {aggregate['neural']['layer_rmse'][field][k]:.6g}")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", f"## Sensitivity to output-logit shifts of +/- {report['logit_shift']:g}", "",
                  "| Perturbation | Loss change vs neural | Standardized forecast-change RMS |",
                  "|---|---:|---:|"])
    for name in aggregate:
        if name in ("physical", "neural"):
            continue
        lines.append(f"| {name} | {aggregate[name]['loss_change_vs_neural']:+.8g} | {aggregate[name]['standardized_forecast_change_rms']:.8g} |")
    lines.extend(["", "Small-sample diagnosis only; no optimization performed. Sensitivity is to logit shifts, not percentage changes in diffusivity."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", default="data/neural_macda")
    parser.add_argument("--output", type=Path, required=True, help="separate diagnosis directory")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--logit-shift", type=float, default=.1)
    args = parser.parse_args()
    if args.examples < 1 or not np.isfinite(args.logit_shift) or args.logit_shift <= 0:
        parser.error("examples and logit-shift must be positive")
    if args.output.resolve() == args.checkpoint.parent.resolve():
        parser.error("use a separate output directory to preserve training logs")
    configure(args.output)
    jax.config.update("jax_enable_x64", True)
    raw = args.checkpoint.read_bytes()
    saved = pickle.loads(raw)  # Only load a checkpoint from your own trusted run.
    contract = saved["contract"]
    if contract.get("version") != 1 or contract.get("features") != list(FEATURE_NAMES):
        raise ValueError("unsupported checkpoint feature/training contract")
    params, mean, scale = saved["params"], saved["mean"], saved["scale"]
    if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((params, mean, scale))) or np.any(np.asarray(scale) <= 0):
        raise ValueError("invalid checkpoint parameters or normalization")
    event("diagnosis_start", checkpoint=str(args.checkpoint), step=saved["completed"],
          device=str(jax.devices()[0]), examples=args.examples)
    experiment = Experiment(contract["max_dt"], contract["horizon"], contract["spinup"])
    warmup = jax.jit(lambda state, context: experiment.advance(state, context, None, experiment.spinup))
    subset, windows = validation_subset(NativeCache(args.cache, contract["revision"]), experiment, warmup, args.examples)

    @jax.jit
    def forecast(parameters, state, context):
        closure = Closure(parameters, mean, scale)
        def interval(carry, _):
            result = experiment.advance(carry, context, closure, 1)
            return result, experiment.fields(result)
        return jax.lax.scan(interval, state, None, length=experiment.horizon)[1]

    physical_params = initialize(jax.random.key(0), constant=contract["constant"])
    variants = {"physical": physical_params, "neural": params}
    for channel, name in enumerate(("momentum", "prandtl")):
        for sign in (-1, 1):
            variants[f"{name}_{sign:+d}"] = shifted(params, channel, sign * args.logit_shift)
    results = {name: [] for name in variants}
    responses = {name: [] for name in variants if name not in ("physical", "neural")}
    samples, active_samples = [], []
    for index, (state, context, target, mass, features) in enumerate(subset, 1):
        factors = np.asarray(multipliers(Closure(params, mean, scale), jnp.asarray(features)))
        samples.append(factors.reshape(-1, 2))
        # These are geometrically eligible interfaces, not a claim of nonzero
        # effective mixing: friction velocity, stability and rate caps also act.
        active_samples.append(factors[np.asarray(features)[..., 10] < experiment.base.pbl_height_m])
        learned = None
        for name, parameters in variants.items():
            with phase("diagnostic_forecast", example=index, total=args.examples, variant=name):
                predicted = np.asarray(forecast(parameters, state, context))
            if not np.isfinite(predicted).all():
                raise ValueError(f"nonfinite diagnostic forecast: {name}")
            metrics = error_metrics(predicted, target, mass)
            results[name].append(metrics)
            if name == "neural":
                learned = predicted
            elif name != "physical":
                responses[name].append(error_metrics(predicted, learned, mass)["loss"])
            event("diagnostic_score", example=index, variant=name, loss=metrics["loss"])
    aggregate = {}
    for name, scores in results.items():
        layer_rmse = np.sqrt(np.mean([r["layer_mse"] for r in scores], axis=0))
        aggregate[name] = {"loss": float(np.mean([r["loss"] for r in scores])),
                           "layer_rmse": {field: layer_rmse[i].tolist() for i, field in enumerate(FIELDS)}}
        if name in responses:
            aggregate[name]["standardized_forecast_change_rms"] = float(np.sqrt(np.mean(responses[name])))
            aggregate[name]["loss_change_vs_neural"] = aggregate[name]["loss"] - aggregate["neural"]["loss"]
    physical_loss = aggregate["physical"]["loss"]
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
              "step": saved["completed"], "contract": contract, "examples": args.examples,
              "windows": windows, "sigma_top_to_bottom": np.asarray(experiment.coords.vertical.centers).tolist(),
              "variants": aggregate, "logit_shift": args.logit_shift,
              "improvement_percent": 100 * (physical_loss - aggregate["neural"]["loss"]) / physical_loss if physical_loss else None,
              "corrections": {},
              "limitations": ["Small fixed validation subset; not an overfit test or held-out-year accuracy claim.",
                              "Correction statistics sample forecast-initial interfaces only; they do not measure effective capped diffusivity.",
                              "Perturbations shift one output logit everywhere, not independently at each interface.",
                              "Layer RMSE uses equal example weights and mass/area weighting within each example."]}
    for region, collection in (("all_interfaces", samples), ("below_pbl_height", active_samples)):
        values = np.concatenate(collection)
        report["corrections"][region] = {name: correction_summary(values[:, i])
                                         for i, name in enumerate(("momentum", "prandtl"))}
    atomic_bytes(args.output / "diagnosis.json", json.dumps(report, indent=2, allow_nan=False).encode())
    atomic_bytes(args.output / "summary.md", render_summary(report).encode())
    event("diagnosis_complete", physical_loss=physical_loss, neural_loss=aggregate["neural"]["loss"],
          improvement_percent=report["improvement_percent"], report=str(args.output / "diagnosis.json"))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        LOGGER.exception("diagnosis_failed")
        raise
