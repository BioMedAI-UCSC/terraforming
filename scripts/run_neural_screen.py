#!/usr/bin/env python3
"""Four short neural methods on four GPUs, paired validation and table logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

from neural_screen_metrics import ensemble_report

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full", "tendency-only", "adapters-only", "decoder-only")
REVISION = "65a0bebd804b9c240752277e83f5737d58c6ee9c"


class Table:
    """Plain ASCII tables work in terminals, redirected logs, and SSH sessions."""
    def __init__(self, streams):
        self.streams = streams
        self.rows = 0
        self.columns = (("UTC", 8), ("GPU", 3), ("METHOD", 13), ("EVENT", 27),
                        ("STEP", 5), ("TRAIN", 10), ("VALID", 10), ("GAIN %", 8),
                        ("SECONDS", 8), ("DETAIL", 48))

    def write(self, line):
        for stream in self.streams:
            print(line, file=stream, flush=True)

    def cells(self, values):
        values = [str(value).replace("\n", " ").replace("\r", " ") for value in values]
        return "| " + " | ".join(value[:width].ljust(width) for value, (_, width) in zip(values, self.columns)) + " |"

    def row(self, method, gpu, event, fields=None):
        fields = fields or {}
        if self.rows % 30 == 0:
            self.write("+-" + "-+-".join("-" * width for _, width in self.columns) + "-+")
            self.write(self.cells([name for name, _ in self.columns]))
        physical, neural = fields.get("physical_loss"), fields.get("neural_loss")
        gain = 100 * (physical - neural) / physical if physical and neural is not None else None
        detail = fields.get("detail") or fields.get("error") or fields.get("path")
        if detail is None:
            keys = ("gradient_norm", "eta_seconds", "latent_loss", "example", "examples", "horizon", "devices", "window", "reason", "includes_jit_compilation")
            detail = " ".join(f"{key}={number(fields[key]) if isinstance(fields[key], float) else fields[key]}" for key in keys if key in fields)
        self.write(self.cells([datetime.now(timezone.utc).strftime("%H:%M:%S"), gpu, method,
                              event, fields.get("step", "-"), number(fields.get("loss")),
                              number(neural), number(gain), number(fields.get("elapsed_seconds")), detail]))
        self.rows += 1

    def comparison(self, report):
        self.write(f"\nPaired validation at step {report['step']} (lower loss is better; equal ensemble weights):")
        self.write("| Method        | Physical   | Validation | Internal   | Gain %     |")
        self.write("|---------------|------------|------------|------------|------------|")
        physical = report["physical_loss"]
        for method, loss, latent in zip((*METHODS, "ensemble"), (*report["member_losses"], report["neural_loss"]),
                                       (*report["member_latent_losses"], None)):
            gain = 100 * (physical - loss) / physical if physical else None
            self.write(f"| {method:13} | {number(physical):>10} | {number(loss):>10} | {number(latent):>10} | {number(gain):>10} |")
        winner = min(zip((*report["member_losses"], report["neural_loss"]), (*METHODS, "ensemble")))[1]
        self.write(f"Lowest decoded loss: {winner}; small validation screen, not an independent test result.")
        self.write("")


def number(value):
    return "-" if value is None else f"{float(value):.6g}"


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/neural_screen/deadline"))
    parser.add_argument("--cache", type=Path, default=Path("data/neural_macda"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument("--mola", type=Path)
    terrain.add_argument("--flat-terrain", action="store_true", help="controlled tests only")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--budget-steps", type=int, default=100, help="fixed curriculum endpoint, retained on resume")
    parser.add_argument("--minutes", type=float, default=30., help="wall-time limit for all four workers, including startup")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--validation-examples", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--normalization-chunks", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spinup", type=int, default=12)
    parser.add_argument("--max-dt", type=float, default=300.)
    parser.add_argument("--refresh-seconds", type=float, default=1800.)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--regularization", type=float, default=.0001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print exact commands without loading JAX or starting workers")
    args = parser.parse_args(argv)
    args.gpus = args.gpus.split(",")
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4 or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("--gpus must name four distinct nonnegative GPU indices")
    counts = (args.steps, args.budget_steps, args.validate_every, args.validation_examples,
              args.horizon, args.chunks, args.normalization_chunks, args.width, args.blocks)
    if min(counts) < 1 or args.spinup < 0 or args.steps > args.budget_steps:
        parser.error("counts must be positive, spinup nonnegative, and steps <= budget-steps")
    if args.spinup + args.horizon >= 120:
        parser.error("soil history plus horizon must fit inside 120 snapshots")
    if any(not math.isfinite(x) or x <= 0 for x in (args.minutes, args.max_dt, args.refresh_seconds, args.learning_rate)):
        parser.error("time limits, timesteps and learning rate must be finite and positive")
    if not math.isfinite(args.regularization) or args.regularization < 0:
        parser.error("regularization must be finite and nonnegative")
    args.output, args.cache = args.output.resolve(), args.cache.resolve()
    if not args.flat_terrain:
        args.mola = (args.mola or Path(os.environ.get("MOLA_PATH", "outputs/nautilus-calibration-inputs/mola.img"))).resolve()
        if not args.dry_run and not args.mola.is_file():
            parser.error(f"MOLA file not found: {args.mola}; set --mola or MOLA_PATH")
    return args


def build_jobs(args):
    jobs = []
    for method, gpu in zip(METHODS, args.gpus):
        folder = args.output / method
        command = [sys.executable, "-u", str(ROOT / "scripts/train_neural_hybrid.py"),
                   "--revision", args.revision, "--cache", str(args.cache), "--output", str(folder),
                   "--devices", "1", "--ablation", method, "--steps", str(args.steps),
                   "--curriculum", f"{args.horizon}:{args.budget_steps}",
                   "--validation-horizon", str(args.horizon), "--validate-every-epochs", "0",
                   "--validate-every-steps", str(args.validate_every), "--export-validation-predictions"]
        for name in ("validation_examples", "chunks", "normalization_chunks", "width", "blocks", "seed",
                     "spinup", "max_dt", "refresh_seconds", "learning_rate", "regularization"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        command += ["--flat-terrain"] if args.flat_terrain else ["--mola", str(args.mola)]
        if args.resume:
            command.append("--resume")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(ROOT / "package") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        jobs.append(dict(method=method, gpu=gpu, folder=folder, command=command, env=env))
    return jobs


def comparison_contract(args):
    omitted = {"output", "gpus", "steps", "minutes", "resume", "dry_run"}
    return dict(schema="four-method-screen-v1", methods=list(METHODS),
                settings={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items() if key not in omitted})


def stop_workers(processes, sig):
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass


def run_jobs(jobs, output, minutes, table, *, grace_seconds=15.):
    """Independent worker failures are reported while the other methods continue."""
    output = Path(output)
    messages = queue.Queue()
    processes, threads, closed = {}, [], set()
    state = {job["method"]: dict(gpu=job["gpu"], status="starting", step=0) for job in jobs}
    ensembles, ensemble_errors = {}, {}
    started = time.monotonic()
    stop_at = None
    timed_out = interrupted = False

    def reader(job, process):
        try:
            with (job["folder"] / "console.log").open("a") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    messages.put((job, line))
        finally:
            process.stdout.close()
            messages.put((job, None))

    def snapshot():
        report = dict(methods=state, ensembles=ensembles, ensemble_errors=ensemble_errors,
                      timed_out=timed_out, interrupted=interrupted,
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "summary.json", report)
        return report

    def collect_ensemble(step):
        key = str(step)
        if key in ensembles or key in ensemble_errors:
            return
        paths = [job["folder"] / "validation-predictions" / f"step-{step:06d}.npz" for job in jobs]
        if not all(path.exists() for path in paths):
            return
        try:
            report = ensemble_report(paths)
            ensembles[key] = report
            atomic_json(output / f"ensemble-step-{step:06d}.json", report)
            table.comparison(report)
        except (ValueError, OSError, KeyError) as error:
            ensemble_errors[key] = str(error)
            table.row("ensemble", "-", "comparison_rejected", {"step": step, "error": str(error)})

    try:
        for job in jobs:
            job["folder"].mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(job["command"], cwd=ROOT, env=job["env"],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       start_new_session=True)
            processes[job["method"]] = process
            state[job["method"]].update(status="running", pid=process.pid)
            thread = threading.Thread(target=reader, args=(job, process), daemon=True)
            thread.start()
            threads.append(thread)
            table.row(job["method"], job["gpu"], "worker_started", {"detail": f"pid={process.pid}; one GPU; independent optimizer"})
        snapshot()
        while len(closed) < len(jobs) or any(process.poll() is None for process in processes.values()):
            now = time.monotonic()
            if stop_at is None and now - started >= minutes * 60 and any(process.poll() is None for process in processes.values()):
                timed_out, stop_at = True, now
                table.row("all", "-", "wall_time_limit", {"elapsed_seconds": now - started})
                stop_workers(processes, signal.SIGINT)
            if stop_at is not None and now - stop_at >= grace_seconds:
                stop_workers(processes, signal.SIGKILL)
            try:
                job, line = messages.get(timeout=.2)
            except queue.Empty:
                continue
            method, gpu = job["method"], job["gpu"]
            if line is None:
                closed.add(method)
                continue
            try:
                fields = json.loads(line)
                if not isinstance(fields, dict) or "event" not in fields:
                    raise ValueError()
            except (ValueError, TypeError):
                table.row(method, gpu, "worker_output", {"detail": line.strip()})
                continue
            event = fields["event"]
            state[method]["last_event"] = event
            if event == "training_progress":
                state[method].update(step=fields["step"], training_loss=fields["loss"], gradient_norm=fields["gradient_norm"])
            elif event == "checkpoint_loaded":
                state[method]["step"] = fields["step"]
            elif event == "small_validation":
                state[method]["validation"] = fields
                state[method]["step"] = fields["step"]
            table.row(method, gpu, event, fields)
            if event == "small_validation":
                collect_ensemble(fields["step"])
                snapshot()
    except KeyboardInterrupt:
        interrupted = True
        table.row("all", "-", "interrupted", {"detail": "Stopping workers; completed checkpoints are retained"})
    finally:
        stop_workers(processes, signal.SIGINT)
        deadline = time.monotonic() + grace_seconds
        for process in processes.values():
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                stop_workers(processes, signal.SIGKILL)
                process.wait()
        for thread in threads:
            thread.join(timeout=2.)
        for method, process in processes.items():
            code = process.returncode
            state[method].update(returncode=code, status="complete" if code == 0 else "stopped" if timed_out or interrupted else "failed")
            table.row(method, state[method]["gpu"], state[method]["status"],
                      {"step": state[method]["step"], "detail": f"exit={code}; raw output: {output / method / 'console.log'}"})
        snapshot()
    return snapshot()


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    if args.dry_run:
        table = Table([sys.stdout])
        for job in jobs:
            table.row(job["method"], job["gpu"], "planned", {"step": args.steps})
            print(f"CUDA_VISIBLE_DEVICES={job['gpu']} JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false " + shlex.join(job["command"]))
        return 0
    # Inventory check avoids launching costly workers with duplicate/missing GPUs.
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                               text=True, capture_output=True, check=True)
    available = {line.strip() for line in inventory.stdout.splitlines()}
    if not set(args.gpus) <= available:
        raise ValueError(f"requested GPUs {args.gpus}; nvidia-smi reports {sorted(available)}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "screen.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another screen runner owns this output directory") from None
        plan_path = args.output / "screen-plan.json"
        contract = comparison_contract(args)
        if args.resume:
            if json.loads(plan_path.read_text())["contract"] != contract:
                raise ValueError("screen comparison contract changed; use a new output directory")
            if not all((job["folder"] / "checkpoint.pkl").exists() for job in jobs):
                raise ValueError("resume requires checkpoints for all four methods; inspect per-method console.log")
        elif any(path.name != "screen.lock" for path in args.output.iterdir()):
            raise ValueError("screen output already exists; use --resume or a new directory")
        plan = dict(contract=contract, target_steps=args.steps, wall_minutes=args.minutes,
                    jobs=[{key: str(value) if isinstance(value, Path) else value
                           for key, value in job.items() if key != "env"} for job in jobs])
        atomic_json(plan_path, plan)
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            with (args.output / "tables.log").open("a") as stream:
                table = Table([sys.stdout, stream])
                report = run_jobs(jobs, args.output, args.minutes, table)
        finally:
            signal.signal(signal.SIGTERM, previous)
        final = report["ensembles"].get(str(args.steps))
        success = (not report["timed_out"] and not report["interrupted"] and not report["ensemble_errors"]
                   and final is not None and all(item["status"] == "complete" for item in report["methods"].values()))
        return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Screen failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
