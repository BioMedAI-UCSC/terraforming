"""GPU vs CPU benchmark for the batched Mars engine (standalone entry point).

 Read to undersand:  
    Time the batched engine on CPU (eager) vs GPU (compiled) per batch size.
    For each B: build B Mars instances on CPU and time an eager run; then, if a
    CUDA device is available and *use_gpu*, build them on the GPU, pay the
    one-time ``torch.compile`` JIT in a warm-up run (measured separately and start a new run), and
    time a steady-state run. Pure logic — no printing — so callers own
    formatting and the result is testable.

    ``on_start(B)`` is called before each batch's work begins and
    ``on_done(row)`` after it completes, so callers can stream progress (the
    GPU compile can take minutes, especially on Windows).

    Returns
    -------
    list[dict]
        One row per batch size with keys ``B``, ``cpu_s``, ``gpu_s``,
        ``jit_s``, ``speedup``, ``status``.  ``gpu_s`` / ``jit_s`` /
        ``speedup`` are ``None`` when the GPU path is skipped or runs out of
        memory.
  WARNING: If you run this on your local GPU, and depending on the size of the batches(the bigger).
  You can expect for some thermal throttling, meaning that the issue is, you laptop compute is
  reaching its limit. You laptop will have some defense against this issue hence, you will see a dip in preformace.

Thin wrapper around the shared benchmark logic in ``cli.runner`` — the same
code that powers ``tform benchmark``. Prefer the CLI for normal use:

    tform benchmark --batch 500,1200,1400 --years 3

This script exists for quick standalone runs:

    uv run python gpu_bench/gpu.py
"""

from cli.runner import run_gpu_benchmark
from cli.models import Accuracy

BATCH_SIZES = [500, 1200, 1400]
DURATION_YEARS = 3
DURATION_SECONDS = 3600.0 * 24 * 365 * DURATION_YEARS
WARMUP_SECONDS = 3600.0 * 24 * 30   # 30 days — enough to trigger the JIT

print(f"Mars Simulation Benchmark — {DURATION_YEARS} year run per simulation")
print(f"{'B':>6}  {'CPU (s)':>10}  {'JIT (s)':>10}  {'GPU (s)':>10}  "
      f"{'Speedup':>10}  Status")
print("-" * 68)

rows = run_gpu_benchmark(BATCH_SIZES, DURATION_SECONDS, WARMUP_SECONDS,
                         accuracy=Accuracy.fast)
for r in rows:
    cpu = f"{r['cpu_s']:>10.2f}"
    jit = f"{r['jit_s']:>10.2f}" if r["jit_s"] is not None else f"{'—':>10}"
    gpu = f"{r['gpu_s']:>10.2f}" if r["gpu_s"] is not None else f"{'—':>10}"
    spd = f"{r['speedup']:>9.2f}x" if r["speedup"] is not None else f"{'—':>10}"
    print(f"{r['B']:>6}  {cpu}  {jit}  {gpu}  {spd}  {r['status']}")
