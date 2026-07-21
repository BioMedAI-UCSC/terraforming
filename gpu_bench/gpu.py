"""
GPU vs CPU benchmark for the Mars terraforming simulation.
Runs B simulations in parallel for a given duration and compares CPU vs GPU time.

   uv run python gpu_bench/gpu.py
"""
import time
import torch
from src.celestials import Mars
from src.engine import BatchedTimeController, Accuracy

# Fails loudly if the chunked-compile fix isn't in the imported package
assert hasattr(BatchedTimeController, "_run_chunked"), \
    "Fix 2 (_run_fast_chunked) not found — check which package install is being imported"

BATCH_SIZES = [500,1200,1400]
DURATION_YEARS = 3
DURATION_SECONDS = 3600.0 * 24 * 365 * DURATION_YEARS
WARMUP_SECONDS = 3600.0 * 24 * 30   # 30 days = 720 steps — enough to trigger the JIT

print(f"Mars Simulation Benchmark — {DURATION_YEARS} year run per simulation")
print(f"{'B':>6}  {'CPU (s)':>10}  {'JIT (s)':>10}  {'GPU (s)':>10}  {'Speedup':>10}  Status")
print("-" * 68)

for B in BATCH_SIZES:
    # ---- CPU baseline (eager) ----
    mars_cpu = [Mars(device='cpu') for _ in range(B)]
    btc_cpu = BatchedTimeController(mars_cpu, accuracy=Accuracy.FAST, compile=False)
    t0 = time.perf_counter()
    btc_cpu.run(DURATION_SECONDS)
    cpu_time = time.perf_counter() - t0
    del mars_cpu, btc_cpu

    # ---- GPU (compiled, chunked) ----
    try:
        torch.cuda.empty_cache()
        mars_gpu = [Mars(device='cuda:0') for _ in range(B)]
        btc_gpu = BatchedTimeController(mars_gpu, accuracy=Accuracy.FAST, compile=True)

        # Warm-up: pays the one-time torch.compile cost for this batch size
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        btc_gpu.run(WARMUP_SECONDS)
        torch.cuda.synchronize()
        jit_time = time.perf_counter() - t0

        # Timed run: steady-state performance
        t0 = time.perf_counter()
        btc_gpu.run(DURATION_SECONDS)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - t0
        speedup = cpu_time / gpu_time
        status = "GPU wins" if gpu_time < cpu_time else "CPU wins"
        print(f"{B:>6}  {cpu_time:>10.2f}  {jit_time:>10.2f}  {gpu_time:>10.2f}"
              f"  {speedup:>9.2f}x  {status}")
        del mars_gpu, btc_gpu
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{B:>6}  {cpu_time:>10.2f}  {'—':>10}  {'OOM':>10}  {'N/A':>10}  {e}")
        torch.cuda.empty_cache()
