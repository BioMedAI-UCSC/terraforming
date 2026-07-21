# Batched Engine — GPU Performance Architecture

## Overview

`BatchedTimeController` runs **B** independent Mars simulations concurrently by
stacking their per-instance state into `[B]`-shaped tensors, so a single kernel
invocation advances all B simulations at once. This document records the
performance redesign that took the batched engine from **losing to the CPU** to
an **11–19× GPU speedup**, the diagnosis that motivated each change, and the
verification and benchmark evidence.

**Primary file:** `package/src/engine/batched_controller.py`
**Reference hardware:** NVIDIA RTX A4000 (workstation, 16 GB, Linux) and NVIDIA
RTX 4060 Laptop (8 GB, Windows). Dtype: float64 (`TF_DTYPE`).

---

## 1. Problem Statement

Batching B simulations into shared `[B]` tensors is intended to raise GPU
occupancy relative to B separate scalar simulations. In practice, the original
implementation was slower than the CPU across all tested batch sizes:

| B   | CPU (s) | GPU (s) | Speedup |
|-----|---------|---------|---------|
| 50  | 137.2   | 152.1   | 0.90×   |
| 100 | 256.4   | 267.4   | 0.96×   |
| 250 | 632.1   | 621.9   | 1.02×   |

The measured cost was dominated by work performed *around* the physics kernels
rather than by the physics itself.

---

## 2. Diagnosis

### 2.1 Per-step object construction (≈80 % of runtime)

A linear fit of runtime versus batch size yields:

```
CPU(B) ≈ 17 s + 2.4 s · B
GPU(B) ≈ 36 s + 2.3 s · B
```

The per-simulation slope is nearly identical on both devices (~2.4 s). A cost
that is invariant across CPU and GPU cannot originate in the physics
computation, whose throughput differs by orders of magnitude between the two
devices; it must originate in device-independent Python execution.

The source was the recording step. The original `_record()` constructed one
`Snapshot` object per simulation per timestep, each performing five per-element
`.clone()` calls:

```python
for i in range(B):
    histories[i].append(Snapshot(
        time=elapsed.clone(),
        surface_temperature=bm._T[i].clone(),
        ...   # five fields
    ))
```

For a 10-year run (~87,600 steps) at B=50 this is ~4.4 million `Snapshot`
objects and ~22 million small GPU copy operations, all serialized per batch
element — the O(1)-per-step batched physics was re-serialized into O(B) work
per step.

### 2.2 Per-step kernel-launch overhead

Each timestep issues ~45 small element-wise CUDA operations (`advance_orbit`
plus the physics update). Kernel-launch overhead is ~5–10 µs per operation on
the reference hardware (higher under the Windows WDDM driver model). Across
~87,600 steps this is tens of seconds of CPU-side dispatch latency during which
the GPU is idle — the actual compute per step occupies well under one second in
aggregate. This fixed per-step cost is independent of B, which is consistent
with the flat GPU timings observed after §2.1 was addressed.

### 2.3 Ineffective compilation

The original code wrapped `compute_derivatives` and `compute_fast_physics`
individually with `torch.compile(mode='reduce-overhead')`. Two factors negated
it:

1. `mode='reduce-overhead'` uses CUDA graphs, which reuse a static output
   memory pool. Because `run()` retains references to every step's state (the
   history), the retained outputs cause PyTorch to fall back from CUDA-graph
   capture without warning.
2. `advance_orbit` was never wrapped, leaving ~8 eager launches per step.

---

## 3. Redesign

### 3.1 `BatchedHistory` — stacked-tensor recording

`run()` returns a `BatchedHistory` dataclass of six stacked tensors: `time` is
`[n_steps]`; every other field is `[n_steps, B]` (rows = timesteps, columns =
simulations).

```python
@dataclass
class BatchedHistory:
    time: torch.Tensor                 # [n_steps]
    surface_temperature: torch.Tensor  # [n_steps, B]
    surface_pressure: torch.Tensor     # [n_steps, B]
    ice_mass: torch.Tensor             # [n_steps, B]
    solar_flux: torch.Tensor           # [n_steps, B]
    orbital_angle: torch.Tensor        # [n_steps, B]
```

Inside the hot loop, references to the current state tensors are appended to
Python lists — no clones and no per-element indexing — then combined with a
single `torch.stack` per field after the loop. This reduces recording from
`5 · B` GPU operations per step to six `torch.stack` operations for the entire
run.

**Correctness invariant.** Recording by reference is valid only because every
physics update reassigns state out of place (`self._T = ...`) rather than
mutating in place. A recorded tensor is therefore never modified by a later
step. Introducing in-place operations (`add_`, `mul_`, …) into `advance_orbit`,
`compute_fast_physics`, or `unpack_state` would silently corrupt recorded
history.

`BatchedHistory.to_lists()` reconstructs the legacy `list[list[Snapshot]]`
format on demand. It is O(B · n_steps) Python object construction by design and
is intended only for small B (e.g. the CLI's 3–4 sites); it must not be invoked
inside a benchmark or a large-B sweep. Result analysis should instead be
vectorized over the stacked tensors:

```python
hist.surface_temperature.mean(dim=0)   # [B] mean temperature per simulation
hist.surface_temperature[:, i]         # full history of simulation i
```

**Effect.** At B=250, a 10-year run fell from 632 s → 17.8 s (CPU) and
622 s → 30.3 s (GPU) — a 20–35× reduction attributable to eliminating per-step
object construction.

### 3.2 `_run_chunked` — chunk-level compilation

When `compile=True` and the device is CUDA, `run()` dispatches to
`_run_chunked`, which advances the simulation in fused blocks of `chunk`
timesteps:

```python
def step_chunk():
    rows = []
    for _ in range(chunk):
        self._evolve(dt)                 # orbit + FAST or RK4
        rows.append(torch.stack([...]))  # [6, B]
    return torch.stack(rows)             # [chunk, 6, B]

step_fn = torch.compile(step_chunk, fullgraph=True)
```

`torch.compile(fullgraph=True)` unrolls the `chunk`-step loop and fuses its
constituent operations into a small number of kernels. For a 10-year run at
`chunk=64` this reduces ~4 million per-operation dispatches to ~1,400 compiled
chunk invocations. `fullgraph=True` requires the entire region to compile,
preventing silent eager fallback (§2.3). Chunk outputs are written into a
pre-allocated `[n_total, 6, B]` buffer; the alternative of concatenating
per-chunk blocks would transiently hold two copies of the history and exceed
device memory at large B.

`mode='reduce-overhead'` is deliberately **not** used here: the retained
history buffer conflicts with CUDA-graph memory pools (§2.3). Default Inductor
mode fuses kernels without that constraint.

**Coverage of both accuracy modes.** `step_chunk` calls `self._evolve`, which
selects the integration strategy:

```python
def _evolve(self, dt):
    self.bmars.advance_orbit(dt)
    if self.accuracy is Accuracy.ACCURATE:
        self._evolve_rk4(dt)                 # 4-sample RK4
    else:
        self.bmars.compute_fast_physics(dt)  # reduced-order update
```

`self.accuracy` is constant for the duration of a run, so the compiler traces a
single branch and fuses whichever strategy was selected. Both FAST and ACCURATE
are therefore served by one compiled path. The RK4 graph is approximately four
times larger, giving a proportionally longer one-time compilation cost (§6).

Dispatch is gated only on the compile flag and device:

```python
if self._compile and self._device.type == 'cuda':
    return self._run_chunked(n_steps, remainder, self._chunk)
```

CPU runs and `compile=False` runs use the record-by-reference loop of §3.1; the
CPU has no kernel-launch overhead for chunk fusion to eliminate.

Steps that do not fill a complete chunk, together with the fractional-`dt`
remainder step, execute eagerly after the compiled loop.

The `__init__` compile block reduces to storing the flag; all prior per-method
`torch.compile` wrappers are removed to avoid nesting compilation inside the
chunk graph:

```python
self._compile = bool(compile)
```

### 3.3 CLI integration

`run_multi` and `run_spots` in `cli/runner.py` consume the legacy format via the
bridge:

```python
all_histories = btc.run(duration=duration).to_lists()
```

B is 3–4 at these call sites, so materialization is inexpensive. Downstream
output (summaries, CSVs, plots) is unchanged.

---

## 4. Verification

An identical run executed with `compile=False` (reference) and `compile=True`
was compared at every timestep, simulation, and field. A 30-sol run (740 steps)
exercises full 64-step chunks, the eager tail, and the remainder step.

| Mode     | First call incl. one-time JIT | Max relative difference vs. reference |
|----------|-------------------------------|----------------------------------------|
| FAST     | 21.6 s                        | T 3.5×10⁻¹⁶ · P 5.7×10⁻¹⁶ · M 0        |
| ACCURATE | 313.0 s                       | T 4.7×10⁻¹⁵ · P 1.5×10⁻¹⁵ · M 1.3×10⁻¹⁶ |

Differences at the 10⁻¹⁶–10⁻¹⁵ level are float64 machine epsilon; they arise
because kernel fusion may reorder floating-point operations and are physically
insignificant. The compiled path reproduces the reference physics to the limit
of float64 precision.

These comparisons are automated in
`package/tests/engine/test_batched_controller.py` for both accuracy modes and a
non-default `chunk`; the CUDA cases are marked `@pytest.mark.slow` and skipped
when no GPU is present.

---

## 5. Performance Results

FAST mode, dt = 3600 s, 3-year runs (26,280 steps), JIT warm-up excluded from
the timed region.

**RTX A4000 (workstation, Linux):**

| B    | CPU (s) | GPU (s) | Speedup |
|------|---------|---------|---------|
| 500  | 13.60   | 1.16    | 11.75×  |
| 1200 | 17.03   | 0.91    | 18.65×  |
| 1400 | 17.72   | 0.91    | 19.41×  |

**RTX 4060 Laptop (Windows):** GPU 0.88–0.97 s; speedups 9.4–11.1×.

GPU runtime is approximately constant across B (≈0.9 s from B=500 to 1400),
indicating the workload remains bounded by fixed per-run costs rather than
element-wise compute at these batch sizes. CPU runtime grows with B. Speedup
ratios are influenced by the CPU baseline (the workstation CPU is slower in
single-thread than the laptop CPU); absolute GPU runtimes are the more stable
comparison. Relative to the original implementation, a 3-year run at B=1400 is
reduced from an estimated ~16 minutes to 0.9 s on the same GPU.

---

## 6. Compilation Cost Model

The first-call times in §4 are dominated by compilation, not simulation:

1. **Compilation (one-time).** The first `run()` call compiles the `chunk`-step
   graph — ~20 s (FAST) or ~5 min (ACCURATE, ~4× larger graph). Incurred once
   per process per batch size; subsequent runs in the same process reuse the
   compiled graph. PyTorch also caches compiled kernels on disk across
   processes.
2. **Simulation.** All steps after the first `chunk`, and all later runs in the
   same process, execute at compiled speed.

Compilation is therefore amortized over long runs (e.g. an 87,600-step run) and
is not justified for short runs, for which `compile=False` avoids the cost.

---

## 7. Configuration Reference

| Setting | Effect |
|---------|--------|
| CPU, any accuracy | Record-by-reference loop (§3.1). Not affected by compilation. |
| GPU, FAST, `compile=True` | Chunked compiled path; ~20 s one-time JIT, then flat in B. |
| GPU, ACCURATE, `compile=True` | Chunked compiled path; ~5 min one-time JIT, then flat in B. |
| GPU, `compile=False` | Record-by-reference loop; appropriate for short runs. |

`chunk` (constructor parameter, default 64, validated ≥ 1) sets the number of
timesteps fused per compiled call. Larger values further amortize
launch/dispatch overhead but increase JIT time and per-chunk memory. Whether an
increase improves runtime depends on whether launch overhead is the dominant
cost; profiling is recommended before tuning.

The compiled path requires Triton, which is bundled with Linux CUDA PyTorch. On
Windows it is provided by the `triton-windows` package (declared as a
platform-gated dependency). If Triton is absent, `compile=True` falls back to
eager execution, observable as a near-zero JIT time on a run that should have
compiled.

---

## 8. Memory and Precision

**History size.** `n_steps × 6 × B × 8` bytes (float64). Example: a 10-year
hourly run (87,600 steps) at B=2000 requires ≈ 8.4 GB. Mitigations for
out-of-memory conditions: reduce B, shorten the duration, or record a subset of
steps.

**Precision.** The compiled path agrees with the eager path to float64 machine
epsilon (§4). Results are not guaranteed bit-for-bit identical because fusion
may reorder arithmetic.

**Reuse semantics.** `BatchedTimeController` and its `BatchedMars` retain and
advance state across `run()` calls; a second `run()` continues from the end of
the first. Independent simulations require freshly constructed `Mars` instances
and controller.

---

## 9. Known Limitations and Future Work

- **Save helper for large runs.** Persisting large histories should use
  `torch.save` of the stacked tensors (binary; ≈4.2 GB for B=2000 × 10 yr) plus
  a per-simulation summary CSV. `to_lists()` must not be used at save time for
  large B.
- **Multi-GPU.** The engine uses a single device. Distributing a batch across
  multiple devices (the reference workstation has four A4000s) offers ~4×
  throughput for large sweeps and is currently reachable only by running one
  process per device with `CUDA_VISIBLE_DEVICES`.
- **Per-B recompilation.** Each new batch size re-incurs the one-time JIT.
  `torch._dynamo.mark_dynamic` on the batch dimension could allow a single
  compilation to serve multiple sizes, at some cost to fusion quality.
- **float32.** The reference GPUs run float64 at ≈1/32 of float32 throughput.
  The flat-in-B timings indicate current runs are bound by fixed overhead and
  memory rather than compute, so a float32 conversion's compute benefit would
  appear chiefly at much larger B; it would still approximately halve memory
  traffic. Deferred.
- **CPU path.** Retains the eager loop by design.
- **`BatchedHistory` export.** Not re-exported from the `src.engine` package
  root; import from `src.engine.batched_controller` when the type is required
  directly.

---

## References and Sources

| Topic | Source |
|-------|--------|
| Kernel-launch overhead exceeding GPU runtime at small batch; CUDA graphs | [PyTorch — Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) |
| CUDA graphs (kernel-launch reduction) | [NVIDIA — Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) |
| `torch.compile` API and modes | [PyTorch — torch.compiler](https://pytorch.org/docs/stable/torch.compiler.html) |
| `torch.compile` usage (incl. `reduce-overhead`) | [PyTorch — Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) |
| Triton (compilation backend) | [Triton language & compiler](https://triton-lang.org/) |
| Triton on Windows (`triton-windows`) | [PyPI](https://pypi.org/project/triton-windows/) · [Repository](https://github.com/woct0rdho/triton-windows) |
| `torch.compile` on Windows (Inductor) | [PyTorch — torch.compile on Windows](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) |
| Dynamic shapes / `mark_dynamic` | [PyTorch — Dynamic shapes](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) |
| FP64 vs FP32 arithmetic throughput (compute capability 8.6) | [NVIDIA — CUDA C++ Programming Guide, Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) |
| Profiling to identify the dominant cost | [PyTorch — torch.profiler](https://pytorch.org/docs/stable/profiler.html) |
| RK4 integration | [Wikipedia — Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) |
