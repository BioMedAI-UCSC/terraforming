#!/usr/bin/env python3
"""Paired equal-start area metrics and whole-temporal-block uncertainty."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from neural_temp_data import digest, read_run, sha256, write_json, write_npz, atomic_bytes
from neural_temp_model import normalized_weights


def metrics(prediction, target, weights, mask=None):
    if prediction.shape != target.shape or weights.shape != target.shape:
        raise ValueError("paired arrays/weights must have identical shape")
    if not all(np.isfinite(a).all() for a in (prediction, target, weights)):
        raise ValueError("nonfinite metric input")
    w = normalized_weights(weights)
    if mask is not None:
        w = w * np.broadcast_to(mask, w.shape)
    sums = w.sum(axis=-1)
    valid = sums > 0
    if not valid.any():
        return {"rmse": None, "mae": None, "bias": None, "starts": 0, "cells": 0, "area_weight_sum": 0.}
    e = (prediction-target)[valid]
    wn = w[valid] / sums[valid, None]
    return {"rmse": float(np.sqrt(np.mean(np.sum(wn*e**2, axis=-1)))),
            "mae": float(np.mean(np.sum(wn*np.abs(e), axis=-1))),
            "bias": float(np.mean(np.sum(wn*e, axis=-1))),
            "starts": int(valid.sum()), "cells": int((w > 0).sum()),
            "area_weight_sum": float(sums.sum()), "per_start_group_weights": sums.tolist()}


def bootstrap(differences, blocks, seed=0, draws=4000):
    """Positive difference means candidate improves MSE; cells never resampled."""
    differences, blocks = np.asarray(differences), np.asarray(blocks)
    unique = np.unique(blocks)
    if len(differences) != len(blocks) or len(unique) < 2:
        raise ValueError("bootstrap requires paired starts in at least two temporal blocks")
    sums = np.array([differences[blocks == b].sum() for b in unique])
    counts = np.array([(blocks == b).sum() for b in unique])
    indices = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    samples = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return {"mean_mse_improvement": float(differences.mean()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(), "seed": seed, "draws": draws,
            "blocks": unique.tolist(), "block_counts": counts.tolist(),
            "block_mean_improvements": (sums/counts).tolist()}


def evaluate(root, split, records, data, predictions):
    root = Path(root)
    target, weights = data["target"], data["area"]
    if len({r["id"] for r in records}) != len(records) or any(r["split"] != split for r in records):
        raise ValueError("unpaired or wrong-split evaluation")
    report = {"split": split, "ids": [r["id"] for r in records], "metrics": {}, "mass_metrics": {},
              "breakdowns": {}, "bootstrap": {}, "metric": "sqrt(mean_start(sum_area(w*error²)))",
              "mass_weight": "start surface pressure × area; fixed sigma thickness cancels",
              "limitations": ["One dataset and one seed; MACDA is reanalysis, not direct truth.",
                             "Postprocessing only: no evidence of improved physics, stability or differentiating through GCM."]}
    season = np.array([r["quadrant"] for r in records])[:, None]
    masks = {f"season_{q}": season == q for q in range(4)}
    for lo, hi in [(-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90)]:
        masks[f"latitude_{lo}_{hi}"] = (data["lat"] >= lo) & ((data["lat"] <= hi) if hi == 90 else (data["lat"] < hi))
    masks.update(day=data["day"].astype(bool), night=~data["day"].astype(bool))
    per_start = {}
    saved = {"target": target, "area": weights, "pressure": data["pressure"],
             "ids": np.array(report["ids"]), "blocks": np.array([r["block"] for r in records])}
    for name, prediction in predictions.items():
        report["metrics"][name] = metrics(prediction, target, weights)
        report["mass_metrics"][name] = metrics(prediction, target, weights*data["pressure"])
        report["breakdowns"][name] = {key: metrics(prediction, target, weights, mask) for key, mask in masks.items()}
        per_start[name] = np.sum(normalized_weights(weights)*(prediction-target)**2, axis=-1)
        saved[f"prediction_{name}"] = prediction
        saved[f"start_mse_{name}"] = per_start[name]
    for control in ("physical", "linear", "no_gcm"):
        delta = per_start[control]-per_start["neural"]
        report["bootstrap"][control] = bootstrap(delta, saved["blocks"])
        saved[f"paired_mse_improvement_vs_{control}"] = delta
        saved[f"paired_squared_error_vs_{control}"] = (predictions[control]-target)**2-(predictions["neural"]-target)**2
    scores = report["metrics"]
    gain = 1-scores["neural"]["rmse"]/scores["physical"]["rmse"]
    report["relative_rmse_reduction"] = gain
    report["criteria"] = {"substantial_gain": gain >= .10,
        "neural_value": scores["neural"]["rmse"] < scores["linear"]["rmse"] and report["bootstrap"]["linear"]["ci95"][0] > 0,
        "physical_forecast_value": scores["neural"]["rmse"] < scores["no_gcm"]["rmse"]}
    report["scientific_success"] = split == "test" and all(report["criteria"].values())
    write_npz(root / f"{split}_predictions.npz", **saved)
    plots(root, split, data, predictions, report)
    write_json(root / f"{split}_report.json", report)
    lines = [f"# Temperature postprocessing: {split}", "", "Lowest sigma midpoint, one native 1/12-sol lead. Equal-start area-weighted metrics.", "",
             "| Model | RMSE K | MAE K | Bias K |", "|---|---:|---:|---:|"]
    for name, score in scores.items():
        lines.append(f"| {name} | {score['rmse']:.4f} | {score['mae']:.4f} | {score['bias']:.4f} |")
    lines += ["", f"Relative neural RMSE reduction: {gain:.2%}.", f"Criteria: {report['criteria']}.",
              f"Held-out scientific success: {report['scientific_success']}.", "",
              "Validation scores are not test results. If the validation gate fails, stop without accessing test forecasts.",
              *report["limitations"], "", "See JSON for all seasonal, latitude and day/night regressions, counts, weights and paired block intervals."]
    atomic_bytes(root / f"{split}_report.md", ("\n".join(lines)+"\n").encode())
    print("model        RMSE K    MAE K    bias K")
    for name, s in scores.items():
        print(f"{name:12} {s['rmse']:8.4f} {s['mae']:8.4f} {s['bias']:9.4f}")
    return report


def plots(root, split, data, predictions, report):
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = root / "plots"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    errors = predictions["neural"]-data["target"]
    sc = axes[0, 0].scatter(data["lon"][0], data["lat"][0], c=errors.mean(axis=0), cmap="RdBu_r", s=10)
    fig.colorbar(sc, ax=axes[0, 0], label="mean temperature error K")
    axes[0, 0].set(xlabel="longitude", ylabel="latitude", title="Neural error map")
    axes[0, 1].hexbin(data["target"].ravel(), predictions["neural"].ravel(), gridsize=50, mincnt=1)
    axes[0, 1].set(xlabel="reference K", ylabel="predicted K")
    axes[1, 0].hexbin(data["lst"].ravel(), errors.ravel(), gridsize=50, mincnt=1)
    axes[1, 0].set(xlabel="local solar time hours", ylabel="prediction residual K")
    for name, result in report["bootstrap"].items():
        axes[1, 1].plot(result["block_mean_improvements"], marker="o", label=name)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set(xlabel="prespecified temporal block index", ylabel="paired MSE improvement K²")
    axes[1, 1].legend()
    fig.savefig(folder / f"{split}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    from cache_neural_temp import load_split
    from train_neural_temp import all_predictions, verify_frozen
    contract, manifest = read_run(args.run_dir)
    frozen = verify_frozen(args.run_dir, contract, require_gate=True)
    seal = args.run_dir / "test_complete.json"
    if seal.exists():
        meta = json.loads(seal.read_text())
        if meta["frozen_hash"] != digest(frozen) or any(sha256(args.run_dir/n) != h for n, h in meta["files"].items()):
            raise ValueError("completed test artifacts changed")
        print("Fixed test already evaluated; verified existing artifacts")
        return
    records, data = load_split(args.run_dir, contract, manifest, "test")
    evaluate(args.run_dir, "test", records, data, all_predictions(args.run_dir, data))
    files = ["test_report.json", "test_report.md", "test_predictions.npz", "plots/test.png"]
    write_json(seal, {"frozen_hash": digest(frozen), "files": {n: sha256(args.run_dir/n) for n in files}})


if __name__ == "__main__":
    main()
