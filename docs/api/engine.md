# Engine API

The `src.engine` module handles numerical integration of planetary state. The integration strategy is fully decoupled from the physics model — `Planet` knows *what* equations to use, `TimeController` knows *how* to integrate them.

Two engines are available:

- **`TimeController`** — drives a single `Planet` through time. Returns `list[Snapshot]`.
- **`BatchedTimeController`** — runs **B** planets in parallel on one device (the GPU path). Returns a `BatchedHistory` of stacked tensors. This is the high-throughput path; see [Batched engine performance](../architecture/batched-engine-performance.md) for the design and benchmarks.

## Accuracy Modes

```python
from src.engine import Accuracy

Accuracy.ACCURATE  # 4th-order Runge-Kutta on the full ODE system
Accuracy.FAST      # Reduced-order analytic / relaxation updates
```

Both modes are supported by both engines and by the compiled batched path.

## Snapshot

The single-simulation engine (`TimeController`) produces one `Snapshot` per timestep. All fields are 0-dim `torch.Tensor` (kept on-device to avoid GPU→CPU syncs); call `.item()` for a Python float.

| Field | Type | Description |
|-------|------|-------------|
| `time` | `torch.Tensor` | Elapsed simulation time in **seconds** |
| `surface_temperature` | `torch.Tensor` | Surface temperature (K) |
| `surface_pressure` | `torch.Tensor` | Surface pressure (Pa) |
| `ice_mass` | `torch.Tensor` | Total polar ice mass (kg) |
| `solar_flux` | `torch.Tensor` | Instantaneous solar flux (W/m²) |
| `orbital_angle` | `torch.Tensor` | True anomaly (rad) |

## BatchedHistory

`BatchedTimeController.run()` returns a `BatchedHistory` instead of per-step objects: recording B × n_steps `Snapshot` objects inside the hot loop was the dominant cost of the batched engine (see the performance doc). Instead, history is six stacked tensors — `time` is `[n_steps]`; every other field is `[n_steps, B]` (rows = timesteps, columns = simulations).

| Field | Shape | Description |
|-------|-------|-------------|
| `time` | `[n_steps]` | Elapsed time in seconds (shared by all B sims) |
| `surface_temperature` | `[n_steps, B]` | Surface temperature (K) |
| `surface_pressure` | `[n_steps, B]` | Surface pressure (Pa) |
| `ice_mass` | `[n_steps, B]` | Total polar ice mass (kg) |
| `solar_flux` | `[n_steps, B]` | Solar flux (W/m²) |
| `orbital_angle` | `[n_steps, B]` | True anomaly (rad) |

**Analyse results vectorised** — one kernel over all B simulations:

```python
hist = btc.run(duration=...)
hist.surface_temperature.mean(dim=0)          # [B] — mean T per simulation
hist.surface_pressure.amax(dim=0)             # [B] — peak P per simulation
hist.surface_temperature[:, 0]                # full history of simulation 0
```

**Legacy format** — `hist.to_lists()` rebuilds the old `list[list[Snapshot]]`
(outer list length B, inner list one `Snapshot` per step). It is O(B × n_steps)
Python object construction *by design*, so only call it for small B (the CLI's
3–4 sites). Never call it inside a benchmark or a large-B sweep.

## Batched engine usage

```python
from src.celestials import Mars
from src.engine import BatchedTimeController, Accuracy

# B parallel simulations; swap 'cuda:0' <-> 'cpu' to pick the device
mars_list = [Mars(latitude=lat, device='cuda:0')
             for lat in (-60, -30, 0, 30, 60)]

btc = BatchedTimeController(
    mars_list,
    dt=3600.0,
    accuracy=Accuracy.FAST,
    compile=True,      # torch.compile-fused chunked loop (CUDA only)
    chunk=64,          # timesteps fused per compiled call (tuning knob)
)
hist = btc.run(duration=668 * 88_775.244)     # 1 Martian year, all B at once
```

### `compile` and `chunk`

- **`compile=True`** (CUDA only): the run executes as
  `torch.compile(fullgraph=True)`-fused blocks of `chunk` timesteps
  (`_run_chunked`), covering both FAST and ACCURATE. The first call pays a
  one-time JIT cost (larger for ACCURATE and for larger `chunk`); subsequent
  runs in the same process reuse the compiled graph. On CPU or with
  `compile=False`, the plain record-by-reference loop runs instead.
- **`chunk`** (default 64): larger values amortise Python/launch overhead
  further but increase JIT time and per-chunk memory. Whether a larger chunk
  helps runtime depends on whether launch overhead is your bottleneck —
  profile before tuning.
- **Triton dependency**: the compiled path needs Triton, bundled with Linux
  PyTorch. On Windows, install `triton-windows`, or `compile=True` falls back
  silently to eager (visible as a near-zero JIT time on a run that should have
  compiled).

### Performance and precision notes

- **float64 throughput**: consumer/workstation GPUs (RTX / A4000-class) run
  float64 at ≈1/32 of their float32 throughput. For maximum speed on such
  hardware, lowering `TF_DTYPE` to float32 is the next major lever (future
  work). Data-centre GPUs (A100, H100) run float64 at ≈1/2 throughput.
- **Memory**: history is `n_steps × 6 × B × 8` bytes (float64) — e.g.
  B=2000 over 10 Earth-years hourly ≈ 8.4 GB. Reduce B, shorten the run, or
  record fewer steps if you hit out-of-memory.
- **Numerical equivalence**: the compiled path agrees with the eager path to
  float64 machine epsilon (≈1e-15 relative); kernel fusion may reorder
  floating-point arithmetic, so results are not guaranteed bitwise identical.

## API reference

::: src.engine
    options:
      members:
        - Accuracy
        - Snapshot
        - TimeController
        - BatchedMars
        - BatchedTimeController
      show_root_heading: true
      show_root_full_path: false

!!! note "`BatchedHistory`"
    `BatchedHistory` is defined in `src.engine.batched_controller` and returned
    by `BatchedTimeController.run()`. It is not currently re-exported from the
    `src.engine` package root; import it from
    `src.engine.batched_controller` if you need the type directly.
