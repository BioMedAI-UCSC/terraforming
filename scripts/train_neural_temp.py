#!/usr/bin/env python3
"""Train six prespecified controls from a frozen cache; never roll out the GCM."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from neural_temp_data import atomic_bytes, digest, read_run, sha256, write_json
from neural_temp_model import (SCHEMA, NO_GCM, ResidualMLP, fit_controls,
    fit_normalization, normalized_weights, predict_neural, standardize)
from cache_neural_temp import load_split


def save_checkpoint(path, state):
    atomic_bytes(path, pickle.dumps(state, protocol=5))


def load_checkpoint(path):
    # Only load trusted artifacts produced by this run; pickle is executable.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def mse(prediction, target, weights):
    return float(np.mean(np.sum(normalized_weights(weights) * (prediction-target)**2, axis=-1)))


def train_model(root, kind, train, validation, contract, *, stop_after=None):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    columns = NO_GCM if kind == "no_gcm" else list(range(len(SCHEMA)))
    base_index = 0 if kind == "no_gcm" else 7
    x, y, w = train["features"], train["target"], train["area"]
    norm = fit_normalization(x, y, x[..., base_index], w, "train")
    model = ResidualMLP(len(columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    folder = Path(root) / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.pkl"
    state = {"kind": kind, "architecture": [len(columns), 32, 32, 1], "activation": "GELU",
             "columns": columns, "normalization": norm, "feature_schema": SCHEMA,
             "contract_hash": digest(contract), "epoch": 0, "updates": 0, "bad_epochs": 0,
             "best_mse": mse(validation["features"][..., base_index], validation["target"], validation["area"])}
    def snapshot():
        state.update(parameters=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                     sampler_state=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state())
        return state
    if path.exists():
        state = load_checkpoint(path)
        if state["contract_hash"] != digest(contract) or state["normalization"] != norm or state["columns"] != columns:
            raise ValueError("training resume contract mismatch")
        model.load_state_dict(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_state"]
        torch.set_rng_state(state["torch_rng"])
    else:
        save_checkpoint(path, snapshot())
        save_checkpoint(folder / "best_validation.pkl", snapshot())
    z = standardize(x, norm, columns)
    probs = normalized_weights(w)
    physical = math.sqrt(mse(validation["features"][..., 7], validation["target"], validation["area"]))
    started = time.monotonic()
    for epoch in range(state["epoch"], 100):
        if state["bad_epochs"] >= 10 or (stop_after is not None and epoch >= stop_after):
            break
        # Shuffle complete starts; every batch includes up to eight distinct starts.
        # Each start contributes 128 area-proportional columns, with NO extra weights.
        order = rng.permutation(len(x))
        for offset in range(0, len(order), 8):
            starts = order[offset:offset+8]
            cells = np.stack([rng.choice(x.shape[1], 128, p=probs[s]) for s in starts])
            s = starts[:, None]
            xb = torch.as_tensor(z[s, cells])
            base = torch.as_tensor(x[s, cells, base_index])
            target = torch.as_tensor(y[s, cells])
            optimizer.zero_grad()
            loss = ((base + model(xb)*norm["residual_rms"] - target)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state["updates"] += 1
        state["epoch"] = epoch + 1
        snapshot()
        val = mse(predict_neural(state, validation["features"]), validation["target"], validation["area"])
        training = mse(predict_neural(state, x), y, w)
        if val < state["best_mse"]:
            state["best_mse"], state["bad_epochs"] = val, 0
            save_checkpoint(folder / "best_validation.pkl", state)
        else:
            state["bad_epochs"] += 1
        # Checkpoint at epoch boundaries includes the exact next sampler state.
        save_checkpoint(path, state)
        event = {"stage": "training", "model": kind, "gpu": "CPU", "epoch": epoch+1,
                 "train_rmse": math.sqrt(training), "validation_rmse": math.sqrt(val),
                 "physical_rmse": physical, "relative_gain": 1-math.sqrt(val)/physical,
                 "elapsed_seconds": time.monotonic()-started, "gpu_hours": 0.}
        with (Path(root) / "training.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"train {kind:8} CPU epoch={epoch+1:3} train={math.sqrt(training):8.4f} val={math.sqrt(val):8.4f} physical={physical:8.4f} gain={event['relative_gain']:7.2%} elapsed={event['elapsed_seconds']:.1f}s GPUh=0", flush=True)
    return load_checkpoint(folder / "best_validation.pkl")


def all_predictions(root, arrays):
    root = Path(root)
    x = arrays["features"]
    params = json.loads((root / "baselines.json").read_text())
    norm = json.loads((root / "normalization.json").read_text())["neural"]
    beta = np.asarray(params["ridge"])
    return {"physical": x[..., 7], "persistence": x[..., 0],
            "constant": x[..., 7]+params["constant"],
            "linear": x[..., 7]+beta[0]+standardize(x, norm) @ beta[1:],
            "neural": predict_neural(load_checkpoint(root / "neural/best_validation.pkl"), x),
            "no_gcm": predict_neural(load_checkpoint(root / "no_gcm/best_validation.pkl"), x)}


def verify_frozen(root, contract, require_gate=False):
    root = Path(root)
    frozen = json.loads((root / "frozen_models.json").read_text())
    if frozen["contract_hash"] != digest(contract):
        raise ValueError("frozen model contract mismatch")
    for name, checksum in frozen["files"].items():
        if sha256(root / name) != checksum:
            raise ValueError(f"frozen artifact changed: {name}")
    if require_gate and not frozen["validation_gate"]:
        raise ValueError("validation gate failed; test is prohibited")
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    contract, manifest = read_run(args.run_dir)
    if (args.run_dir / "frozen_models.json").exists():
        verify_frozen(args.run_dir, contract)
        print("Models already frozen; no refitting")
        return
    _, train = load_split(args.run_dir, contract, manifest, "train")
    records, validation = load_split(args.run_dir, contract, manifest, "validation")
    norms = {kind: fit_normalization(train["features"], train["target"], train["features"][..., index], train["area"], "train")
             for kind, index in (("neural", 7), ("no_gcm", 0))}
    write_json(args.run_dir / "normalization.json", norms)
    write_json(args.run_dir / "baselines.json", fit_controls(train["features"], train["target"], train["features"][..., 7], train["area"], norms["neural"], "train"))
    for kind in ("neural", "no_gcm"):
        train_model(args.run_dir, kind, train, validation, contract)
    from evaluate_neural_temp import evaluate
    report = evaluate(args.run_dir, "validation", records, validation, all_predictions(args.run_dir, validation))
    scores = report["metrics"]
    gate = bool(scores["neural"]["rmse"] <= .9*scores["physical"]["rmse"] and scores["neural"]["rmse"] < scores["linear"]["rmse"])
    files = ["normalization.json", "baselines.json", "validation_report.json",
             "neural/best_validation.pkl", "no_gcm/best_validation.pkl"]
    write_json(args.run_dir / "frozen_models.json", {"contract_hash": digest(contract),
        "validation_gate": gate, "files": {name: sha256(args.run_dir / name) for name in files}})
    print(f"Validation gate: {'PASS — eligible for fixed test' if gate else 'STOP — substantial improvement not demonstrated; no test evaluation'}")


if __name__ == "__main__":
    main()
