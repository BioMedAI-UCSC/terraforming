# Neural PBL training

The optional `neural_closure` argument on both coupled equation builders reaches
`pbl_vertical_diffusion_tendencies`. A 13 → 32 → 32 → 2 tanh MLP multiplies
momentum diffusivity and turbulent Prandtl number by
`exp(log(2) * tanh(output))`, bounded to [0.5, 2]. The output layer is zero at
initialization, giving exactly unit corrections. Physical stability factors,
momentum rate caps, the conservative implicit column solve and dissipative
heating still apply. Omitting the closure preserves the original path.

The 13 SI inputs are upper/lower temperature, upper/lower zonal wind,
upper/lower meridional wind, upper/lower potential temperature, friction
velocity, Richardson number, interface height, layer separation and upper
sigma. These are project feature choices; no paper feature specification was
provided. Normalization samples only training snapshots, then freezes the
statistics in the checkpoint. `--constant` replaces the MLP with two trainable
constant logits for a separate baseline run.

## Four GPUs working together

Use a Linux CUDA host with four GPUs and a CUDA-enabled JAX installation
compatible with Dinosaur. Install the project's `gcm3d` and `arco` extras.
The launcher runs **one process and one optimizer**. Each GPU receives a
different trajectory. `jax.pmap` differentiates each local loss, `lax.pmean`
averages gradients, and every replica applies the same clipped Adam update.
Global batch size is four (one example per GPU). This supports one host;
multi-host JAX initialization is not implemented.

From the repository root:

```bash
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --steps 1000
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --steps 2000 --resume
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --evaluate validation --chunks 1
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --evaluate test
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --constant --output outputs/neural_pbl/constant
```

Evaluation requires the same training contract flags as its checkpoint. The
evaluation `--chunks` limit can be set independently of the training subset.
Validation is MY32–33, test MY34–35, and the separate `storm` split is MY28.
Only MY24–27 and MY29–31 participate in fitting. MY36 is excluded.

Stage a bounded native-cadence sample, or run a CPU smoke test:

```bash
rtk proxy .venv/bin/python scripts/train_neural_pbl.py \
  --revision 65a0bebd804b9c240752277e83f5737d58c6ee9c \
  --output outputs/neural_pbl/staging --stage train --chunks 1

rtk proxy .venv/bin/python scripts/train_neural_pbl.py \
  --revision 65a0bebd804b9c240752277e83f5737d58c6ee9c \
  --output outputs/neural_pbl/smoke --devices 1 --cpu --steps 1 \
  --spinup 0 --chunks 1 --normalization-chunks 1
```

## Data and rollout conventions

### Small checks before and during training

Training now checks a fixed four-example MY32–33 holdout before any optimizer
updates, every epoch by default, and at the final requested step. The small
holdout uses two chunks spread through the validation chronology. It never
contributes gradients or normalization statistics. Physical reference scores
and warmed-up initial states are computed once per invocation and reused.
This is a monitoring sample, not a substitute for full held-out evaluation.

Run just that check with the production initialization and no optimization:

```bash
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh --preflight-only
```

For a cheaper numerical check, use a separate output directory and reduced
sampling/warmup. This does not validate the production soil warmup:

```bash
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh \
  --output outputs/neural_pbl/quickcheck --preflight-only \
  --spinup 0 --normalization-chunks 1 --chunks 1 --validation-examples 2
```

Validate every three epochs and also every 100 updates:

```bash
rtk proxy bash scripts/launch_neural_pbl_4gpu.sh \
  --validate-every-epochs 3 --validate-every-steps 100
```

An epoch is one complete pass through the selected training windows, dropping
the incomplete final global batch. Epoch checks use completed optimizer steps
and preserve their schedule on resume. `--chunks` changes the training subset
and thus epoch length. A full-data epoch can exceed the default 1,000-update
run, so step-based checks are useful. Setting either interval to zero disables
that schedule; initial and final checks remain enabled.

`small_validation.json` contains the latest physical/learned forecast losses,
step, fractional epoch, and exact source windows. `validation.jsonl` records
the history. `best_validation.pkl` retains the lowest-loss monitored checkpoint,
including the initial physical-equivalent model if training does not improve
on it. It includes the optimizer and data position for resumption. The normal
`checkpoint.pkl` continues to track the latest update. Nonfinite validation
stops the run; validation does not currently trigger early stopping for a finite
but worsening loss. Monitoring frequency and sample count can change on resume.

The full Hugging Face commit SHA above was resolved on 2026-09-21. The loader
fetches contiguous chunks of at most 120 snapshots, keeps all 12 snapshots per
sol, and interpolates to T21/L12. Chunks overlap by one snapshot and never cross
Mars-year boundaries. Forecast windows stay inside a chunk; chunk-edge windows
without sufficient warmup/forecast history are omitted. Cache identity includes
revision, grid and preprocessing version. File locks, atomic replacements and
SHA-256 manifests protect concurrent access. A single loader thread prefetches
one upcoming chunk while the training process works on the current chunk.
GPU replicas share the batch loader, so they do not download separately.

Atmospheric temperature and winds are spectrally projected, surface pressure is
converted to nondimensional log pressure, and frost mass becomes pressure
equivalent. Terrain is flat in this first training harness. Dust is held at the
initial observed column value for each trajectory, with longwave opacity set
to visible/3. The solar-longitude phase and fractional-sol diurnal time are
initialized from the source coordinates. The existing solar-angle convention
treats integer sols as noon at zero longitude; the archive's absolute local-time
phase has not independently been verified. Check that convention before a long
fit so a phase mismatch does not become a learning target. The physical radiation, convection,
surface, soil and CO₂ operators are unchanged.

The timestep is `native_interval / ceil(native_interval / max_dt)`, about 296 s
for the default 300 s maximum. Every forecast endpoint lands exactly on a
native timestamp. All soil layers initially equal the observed surface
temperature. A default 12-snapshot (one-sol) **physical** coupled warmup evolves
the whole state before either forecast branch. No soil observations or future
atmospheric values enter initialization. This is a documented initialization
assumption, not an equilibrated deep-soil state. Compare warmup lengths before
interpreting climate skill. `--spinup 0` is for cheap smoke tests.

Training compares every forecast endpoint, using Gaussian area and target
pressure/sigma layer-mass weights. Temperature and both winds use fixed scales
of 10 K and 10 m/s. The loss adds squared log-multiplier regularization sampled
at the forecast's initial interfaces. Physical warmup is outside the gradient;
the complete coupled forecast is inside reverse-mode AD with rematerialized
steps. Only closure parameters receive optimizer updates.

`physical_preflight.json` records the physical forecast error before the first
optimization step; nonfinite forecasts stop training. Review its error before
a long run: passing this numerical gate alone does not establish a scientifically
adequate initialization. Training logs include loss and the global gradient
norm before clipping. Every update atomically saves parameters, Adam moments,
step count, frozen statistics, data cursor, shuffled chunk order, RNG state and
the run contract. Resume rejects incompatible settings and skips consumed
examples. Checkpoints use pickle and should only be loaded from trusted runs.

Evaluation reports mean standardized mass-weighted forecast loss for the
physical and learned closures on identical initial states. It is a reanalysis
comparison, not an observational accuracy claim. Cite ARCO DOI
10.57967/hf/8771 and MACDA DOI 10.5285/cd037a9ea387438fabf4d674dbe53088.

## Verification

```bash
rtk proxy .venv/bin/python -m pytest \
  package/tests/gcm3d/test_neural_pbl.py \
  package/tests/gcm3d/test_neural_parallel.py -q
```

Tests cover identity and disabled behavior, bounds, conservative column
exchange, native cadence, split rejection, cache corruption, coupled-rollout
reverse-mode finite differences, finite neural updates, and four virtual CPU
devices matching a single global-batch update including optimizer resumption.
Virtual devices verify collective semantics, not CUDA/NCCL behavior or GPU
memory requirements. Actual four-GPU throughput and long-run stability must
be measured on the target host.
