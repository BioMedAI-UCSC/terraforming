#!/usr/bin/env python3
"""Four masked GPU cache workers, conservative accounting, validation-gated test.

Metadata staging runs separately on CPU with neural_temp_data.py. Training uses
CPU, leaving GPUs unreserved. Interrupted reservations are charged in full.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from neural_temp_data import read_run, write_json, sha256, digest, atomic_bytes
from cache_neural_temp import read_entry

CAPS = {"train_validation": 2.25, "training": .25, "test": 1., "contingency": .5}


def device_environments(devices):
    if len(devices) != 4 or len(set(devices)) != 4 or any(not str(d).isdigit() for d in devices):
        raise ValueError("exactly four unique numeric GPU IDs required")
    return [{**os.environ, "CUDA_VISIBLE_DEVICES": str(d), "NEURAL_TEMP_GPU": "1",
             "JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "true",
             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"} for d in devices]


def stop_workers(workers):
    for process in workers:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic()+5
    for process in workers:
        if process.poll() is None:
            try:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def supervise(commands, environments, logs, timeout, progress=None, on_spawn=None):
    """Propagate any failed worker and terminate siblings; preserve all logs."""
    workers, handles = [], []
    started = time.monotonic()
    try:
        for command, env, log in zip(commands, environments, logs, strict=True):
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            handle = open(log, "ab", buffering=0)
            handles.append(handle)
            workers.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
            if on_spawn:
                on_spawn([p.pid for p in workers])
        last = 0
        while True:
            codes = [p.poll() for p in workers]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f"worker failed: {codes}; see {logs}")
            elapsed = time.monotonic()-started
            if elapsed >= timeout:
                raise TimeoutError(f"stage timeout after {elapsed:.1f}s; cache progress preserved")
            if progress and (elapsed-last >= 10 or all(c is not None for c in codes)):
                progress(elapsed)
                last = elapsed
            if all(c == 0 for c in codes):
                break
            time.sleep(.2)
    finally:
        stop_workers(workers)
        for handle in handles:
            handle.close()
    return time.monotonic()-started


def remaining(ledger, stage):
    spent = sum(r["charged_gpu_hours"] for r in ledger)
    stage_spent = sum(r["charged_gpu_hours"] for r in ledger if r["stage"] == stage)
    return max(0., min(4.-spent, CAPS[stage]-stage_spent))


def cache_stage(root, records, contract, devices, stage, ledger, timeout=None):
    missing = [r for r in records if read_entry(root, r, contract) is None]
    if not missing:
        return 0.
    envs = device_environments(devices)
    shards = [missing[i::4] for i in range(4)]
    count = sum(bool(s) for s in shards)
    available = remaining(ledger, stage)
    # Keep termination/reaping inside the allocation even on worker failure.
    seconds = max(0., available*3600/count - 6.)
    if timeout is not None:
        seconds = min(seconds, timeout)
    if seconds <= 0:
        raise RuntimeError(f"{stage} GPU budget exhausted; no further work launched")
    reservation = {"stage": stage, "workers": count, "status": "reserved",
                   "charged_gpu_hours": (seconds+6)*count/3600, "started_utc": time.time()}
    ledger.append(reservation)
    write_json(root / "budget.json", ledger)  # Fail closed if parent crashes.
    commands, environments, logs = [], [], []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        commands.append([sys.executable, str(Path(__file__).with_name("cache_neural_temp.py")),
                         "--run-dir", str(root), "--ids", *[r["id"] for r in shard]])
        environments.append(envs[i])
        logs.append(root / "logs" / f"{stage}_{len(ledger)}_gpu{devices[i]}.log")
    def progress(elapsed):
        done = sum((root / "cache" / r["id"] / "complete.json").exists() for r in records)
        event = {"stage": stage, "completed": done, "total": len(records), "gpu": devices,
                 "elapsed_seconds": elapsed, "gpu_hours": sum(r["charged_gpu_hours"] for r in ledger[:-1])+elapsed*count/3600}
        write_json(root / "progress.json", event)
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(event)+"\n")
        print(f"{stage:18} GPUs={','.join(devices)} starts={done}/{len(records)} elapsed={elapsed:.1f}s GPUh={event['gpu_hours']:.4f}", flush=True)
    started = time.monotonic()
    def record_pids(pids):
        reservation["pids"] = pids
        write_json(root / "budget.json", ledger)
    try:
        supervise(commands, environments, logs, seconds, progress, record_pids)
        if any(read_entry(root, r, contract) is None for r in records):
            raise RuntimeError("worker exited without all requested cache artifacts")
        reservation["status"] = "complete"
    except BaseException:
        reservation["status"] = "failed"
        raise
    finally:
        reservation["charged_gpu_hours"] = (time.monotonic()-started)*count/3600
        write_json(root / "budget.json", ledger)
    return reservation["charged_gpu_hours"]


def reconcile_reservations(ledger):
    """Resume after a hard parent kill only when every recorded worker is dead.

    Unknown/PID-reused/live workers fail closed. Full original charges remain;
    already committed entries are still reusable even if no budget remains.
    """
    for reservation in ledger:
        if reservation["status"] != "reserved":
            continue
        if len(reservation.get("pids", [])) != reservation["workers"]:
            raise RuntimeError("interrupted launch with unknown worker PIDs; inspect orphan workers before a new bounded run")
        for pid in reservation["pids"]:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f"interrupted worker PID {pid} may still be alive; no new workers launched")
        reservation["status"] = "interrupted_full_charge"


def run(root, devices, stage):
    contract, manifest = read_run(root)
    if (root / "budget.json").exists():
        ledger = json.loads((root / "budget.json").read_text())
    else:
        ledger = []
    reconcile_reservations(ledger)
    write_json(root / "budget.json", ledger)
    if stage in ("test", "report"):
        from train_neural_temp import verify_frozen
        frozen = verify_frozen(root, contract)
        if not frozen["validation_gate"]:
            write_json(root / "result.json", {"status": "validation_gate_failed", "scientific_success": False,
                       "test_evaluated": False, "gpu_hours": sum(r["charged_gpu_hours"] for r in ledger)})
            print("STOP: substantial improvement not demonstrated on validation. Test not launched.")
            finalize(root, contract, manifest, tested=False)
            return
        if stage == "test":
            tests = [r for r in manifest["starts"] if r["split"] == "test"]
            pilot = json.loads((root / "pilot.json").read_text())
            missing = sum(read_entry(root, r, contract) is None for r in tests)
            if pilot["gpu_hours_per_start"]*missing*1.25 > remaining(ledger, "test"):
                raise RuntimeError("projected test workload exceeds remaining evaluation reserve")
            cache_stage(root, tests, contract, devices, "test", ledger)
            write_cache_index(root, contract, manifest, tested=True)
            print("Test forecasts complete. Release GPUs, then run --stage report on CPU.")
            return
        cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
        supervise([[sys.executable, str(Path(__file__).with_name("evaluate_neural_temp.py")), "--run-dir", str(root)]],
                  [cpu_env], [root / "logs/evaluation.log"], timeout=600)
        report = json.loads((root / "test_report.json").read_text())
        write_json(root / "result.json", {"status": "complete", "scientific_success": report["scientific_success"],
                   "test_evaluated": True, "gpu_hours": sum(r["charged_gpu_hours"] for r in ledger)})
        finalize(root, contract, manifest, tested=True)
        return
    training = [r for r in manifest["starts"] if r["split"] != "test"]
    missing = [r for r in training if read_entry(root, r, contract) is None]
    pilot_path = root / "pilot.json"
    if not pilot_path.exists() and missing:
        pilot = missing[:4]
        hours = cache_stage(root, pilot, contract, devices, "train_validation", ledger, timeout=600)
        # Includes startup, data loading, compilation and reserved idle time.
        per_start = hours/len(pilot)
        estimate = per_start*sum(read_entry(root, r, contract) is None for r in training)*1.25
        test_estimate = per_start*manifest["requested"]["test"]*1.25
        write_json(pilot_path, {"gpu_hours_per_start": per_start, "remaining_train_estimate": estimate,
                               "test_estimate": test_estimate, "safety_factor": 1.25})
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        estimate = pilot["gpu_hours_per_start"]*sum(read_entry(root, r, contract) is None for r in training)*1.25
        if estimate > remaining(ledger, "train_validation") or pilot["test_estimate"] > CAPS["test"]:
            raise RuntimeError(f"projected workload exceeds budget: train remaining {estimate:.3f} GPUh, test {pilot['test_estimate']:.3f} GPUh; propose revised scope before continuing")
    cache_stage(root, training, contract, devices, "train_validation", ledger)
    write_cache_index(root, contract, manifest, tested=False)
    print("Training/validation forecasts complete. Release GPUs, then run train_neural_temp.py on CPU.")


def write_cache_index(root, contract, manifest, tested):
    index = {}
    for record in manifest["starts"]:
        if record["split"] == "test" and not tested:
            continue
        if read_entry(root, record, contract) is None:
            raise ValueError("incomplete final cache")
        index[record["id"]] = json.loads((root / "cache" / record["id"] / "complete.json").read_text())
    write_json(root / "cache_index.json", index)


def finalize(root, contract, manifest, tested):
    write_cache_index(root, contract, manifest, tested)
    required = ["contract.json", "manifest.json", "normalization.json", "baselines.json", "frozen_models.json",
                "training.jsonl", "validation_report.json", "validation_report.md", "validation_predictions.npz",
                "plots/validation.png", "neural/checkpoint.pkl", "neural/best_validation.pkl",
                "no_gcm/checkpoint.pkl", "no_gcm/best_validation.pkl", "cache_index.json", "result.json"]
    if tested:
        required += ["test_complete.json", "test_report.json", "test_report.md", "test_predictions.npz", "plots/test.png"]
    report = root / ("test_report.md" if tested else "validation_report.md")
    atomic_bytes(root / "report.md", report.read_bytes())
    write_json(root / "artifacts.json", {"contract_hash": digest(contract), "test_evaluated": tested,
               "files": {name: sha256(root / name) for name in [*required, "report.md"]}})
    print(f"Artifacts verified; report: {root / 'report.md'}")


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}; stopping workers and preserving progress")
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    p.add_argument("--stage", choices=["train-validation", "test", "report"],
                   help="GPU cache stages exit before CPU training/reporting so allocations can be released")
    p.add_argument("--verify", action="store_true", help="verify an existing complete/gate-stopped run without computing")
    args = p.parse_args()
    root = args.run_dir.resolve()
    if args.verify:
        contract, manifest = read_run(root)
        artifacts = json.loads((root / "artifacts.json").read_text())
        if artifacts["contract_hash"] != digest(contract) or any(sha256(root/n) != h for n, h in artifacts["files"].items()):
            raise ValueError("artifact checksum mismatch")
        from train_neural_temp import verify_frozen
        verify_frozen(root, contract, require_gate=artifacts["test_evaluated"])
        for r in manifest["starts"]:
            if (r["split"] != "test" or artifacts["test_evaluated"]) and read_entry(root, r, contract) is None:
                raise ValueError("missing cache")
        print(json.dumps(json.loads((root / "result.json").read_text()), indent=2))
        return
    if args.stage is None:
        p.error("--stage is required unless --verify is used")
    device_environments(args.gpus)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, args.gpus, args.stage)


if __name__ == "__main__":
    main()
