# Temperature-only neural experiment: launch and acceptance

This branch implements `neural-temp-only-plan.md`. Software verification and
held-out scientific success are separate. The local synthetic integration test
does **not** establish a Mars forecast improvement. Do not reuse the earlier
four-case screen as a baseline.

## Verify the implementation

Run from the repository root:

```bash
rtk proxy uv sync --package terraforming --extra gcm3d --extra arco --dev
rtk proxy .venv/bin/python -m pytest package/tests/gcm3d/test_neural_temp.py -q
```

Expected: **19 passed** when the local native MACDA smoke-test input and MOLA
are staged. Otherwise the real-forecast smoke test is explicitly skipped; the
synthetic tests still exercise all six controls, full 112/32/32 cache coverage,
training, freezing, evaluation, plots, checksums and repeat-command behavior.
The optional native smoke input is
`data/neural_macda/65a0bebd804b9c240752277e83f5737d58c6ee9c/native-t21-l12-v1/000000-000120.nc`.
It supplies observations only, never old forecast arrays.

Focused verification covers out-of-order source blocks, gaps, year/split
boundaries, deterministic and disjoint seasonal selection, invalid units,
training-only statistics, weighted ridge/constant controls, zero-initialized
MLP identity, finite gradients, nonlinear learning, exact epoch-boundary resume,
removal of all physical forecast inputs from the ablation, observed spectral
initialization, future-predictor isolation, soil conduction, paired weighted
metrics, temporal bootstrap units, corruption rejection, worker failures and
timeouts. The integration fixture deliberately uses synthetic temperatures.

## Prepare the actual experiment on CPU

Use the same checkout, dependency versions, filesystem paths and shared output
directory for every stage. The contract pins package versions, scientific
source hashes, Git revision, native archive revision, terrain checksum,
manifest, feature schema, physical configuration and fitting settings.
Scientific code/dependency changes require a new run directory.

For the GPU host, install the CUDA plugin matching the locked JAX version and
its supported driver **before creating the contract**. For a CUDA 12 host:

```bash
rtk proxy uv pip install 'jax[cuda12]==0.11.0'
rtk proxy .venv/bin/python -c 'import jax; print(jax.devices())'
```

Expected on the GPU host: four or more CUDA devices. A CPU-only JAX installation
is sufficient for tests, staging, fitting and reporting, but cache workers
explicitly reject CPU fallback. Use the `.venv/bin/python` commands below after
installing the plugin; an extra dependency synchronization can remove it.

Stage MOLA if absent, then prepare the manifest without reserving GPUs:

```bash
rtk proxy .venv/bin/python scripts/stage_mola.py
rtk proxy env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python scripts/neural_temp_data.py --run-dir outputs/neural_temp/temp_only_v1 --mola data/mola/meg004/megt90n000cb.img
```

Expected: `Frozen native coverage: {'train': 112, 'validation': 32, 'test': 32}`.
`manifest.json`, `contract.json`, `coverage.json`, and 176 checked native windows
appear. Metadata selection uses actual coordinates and a fixed seed. Four
22.5-degree strata within each seasonal quadrant distribute its four dates.
Every selected one-sol-history/start/target window has native cadence, finite
fields, correct units, one split and disjoint intervals. A deficit returns
nonzero and records its reasons; do not shrink the experiment to make it pass.
Network or staging errors also fail rather than becoming data exclusions.

The source is pinned to
`65a0bebd804b9c240752277e83f5737d58c6ee9c`. MACDA's CF `days since` encoding is
interpreted as native Martian sols only with its documented `calendar=none`.
The global assimilation attribute is not attributed to individual windows;
per-window provenance remains explicitly unknown. Native data transfers can be
large because source chunks contain more than each selected 14-sample window.

## Cache training/validation, then release GPUs

```bash
rtk proxy .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --stage train-validation --gpus 0 1 2 3
```

Expected: unique one-device worker masks, device checks in `logs/`, progress
tables, JSONL events, 144 committed cache entries and `cache_index.json`.
The first four requested forecasts form the timing pilot; their entries are
retained. Compilation, preparation inside workers and idle time until the last
worker finishes count toward aggregate GPU-hours. The launcher projects the
remaining cost with a 25% margin and refuses an over-budget campaign before
spending the test reserve. This refusal is an operational stop, not evidence
about skill.

The budgets are 2.25 GPU-hours for training/validation forecasts, 1.00 for test
forecasts, 0.25 reserved for fitting (unused because fitting is CPU-only), and
0.50 contingency (not silently spent). Four GPUs for one hour are four
aggregate GPU-hours. `budget.json` preserves charges across restarts. The
launcher masks processes; it does not allocate/release a cloud instance or a
scheduler job. Allocate GPUs only for these cache commands and release them
on exit. Do not keep a four-GPU job reserved during CPU staging or regression;
external reservation time must also remain within the four-hour overall cap.

Each forecast uses T21/L12, float64, MOLA, the base radiation/CO2 solver,
positivity projection, PBL diffusion, regolith, stability exchange and convective
adjustment. The timestep divides one native interval into 25 steps, each about
295.91748 seconds. Start dust stays fixed (longwave opacity is visible/3).
The solver clock includes the MACDA-midnight/solver-noon half-sol offset and
the eccentric orbit is anchored to start Ls. Regolith starts uniformly at the
oldest available surface temperature, then uses backward-Euler finite-volume
conduction with 12 past, piecewise-constant boundary intervals and zero bottom
flux. No future surface, wind, pressure or dust is used as a predictor.

## Fit and inspect the validation gate on CPU

After releasing the GPU allocation:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python scripts/train_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1
rtk read outputs/neural_temp/temp_only_v1/frozen_models.json
rtk read outputs/neural_temp/temp_only_v1/validation_report.md
```

Expected: both 32/32 GELU models, training-only normalizers, constant and ridge
parameters, `checkpoint.pkl` and `best_validation.pkl` for each model,
`training.jsonl`, six-model validation metrics, prediction arrays and plots.
Training is Adam with learning rate .001, weight decay .0001, gradient clipping
1, seed 0, at most 100 epochs, patience 10. Validation includes the original
zero-correction checkpoint. Each batch samples 128 area-proportional columns
from each of eight shuffled starts; no second area weighting is applied.

`validation_gate: true` requires **at least 10% RMSE reduction versus physical
and lower RMSE than ridge**. If false, do not launch test caching. Finish the
gate-stopped report using the CPU `--stage report` command below. This is a
valid completed experiment with `scientific_success: false` and no test score.
There is no automatic architecture sweep or test-driven tuning.

## Fixed test, only after the gate passes

Acquire the four GPUs only for this command, then release them:

```bash
rtk proxy .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --stage test --gpus 0 1 2 3
```

Expected: 32 additional test cache entries, all tied to the existing frozen
models. The worker and runner both verify the validation gate and frozen
checksums. Repeating a cache command reuses completed matching entries and
simulates missing starts only.

Generate the final report and verify all artifacts on CPU:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --stage report
rtk proxy .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --verify
rtk read outputs/neural_temp/temp_only_v1/report.md
rtk read outputs/neural_temp/temp_only_v1/test_report.json
```

The last command applies only when test was evaluated. Expected verification
output for **scientific success**:

```json
{
  "status": "complete",
  "scientific_success": true,
  "test_evaluated": true
}
```

The output also includes numeric `gpu_hours`, which must be no greater than 4.
For a machine-checkable scientific acceptance command (after `--verify`):

```bash
rtk proxy .venv/bin/python -c 'import json; r=json.load(open("outputs/neural_temp/temp_only_v1/result.json")); assert r["status"] == "complete" and r["test_evaluated"] and r["scientific_success"] and r["gpu_hours"] <= 4, r; print("PASS: fixed held-out criteria and GPU budget")'
```

Expected on scientific success: `PASS: fixed held-out criteria and GPU budget`.
A failed gate or failed test criterion must make this acceptance command fail.

The primary metric is `sqrt(mean_start(sum_cell(area * error²)))` in kelvin.
Success requires all three prespecified test criteria:

1. Neural RMSE is at most 90% of paired physical RMSE.
2. Neural RMSE beats fitted ridge and the 95% paired temporal-block interval
   for `linear MSE - neural MSE` is entirely above zero.
3. Neural RMSE beats the matched no-GCM model, which predicts from start T
   without any forecast inputs or physical additive term.

Otherwise report the failed criterion; command success or training loss is not
scientific success. `test_report.json` contains all seasonal, latitude and
day/night results, including regressions, with counts and weights. It also has
secondary pressure-times-area metrics (start pressure, equal start weight),
paired bootstrap definitions (Mars year × quadrant), seed 0 and 4,000 draws.
Each bootstrap draw resamples whole blocks, never grid cells. Saved predictions
include per-start MSEs, paired squared-error arrays and paired MSE improvements.
Plots include error maps, predicted/reference temperatures, residual/local-time
and paired block skill. One dataset and seed remain limited evidence.

`artifacts.json` and `test_complete.json` seal report/model/prediction checksums.
Repeated completed test commands verify and reuse the existing result; they do
not refit. Pickled checkpoints must be treated as trusted local artifacts only.

## Failures and resume

Repeat the same stage and run directory after an ordinary worker failure,
timeout or handled interruption. Cache files use atomic writes and a final
checksum commit marker. Training resumes at its last complete epoch with model,
optimizer, update count and sampler state. No completed physical jobs need to
be repeated. Contract mismatches and corrupt completed artifacts fail loudly.

A hard kill of the parent can leave an unresolved reservation and orphan GPU
processes. The budget is conservatively charged in full. When recorded workers
are all dead, resume reconciles the reservation and reuses completed entries;
additional jobs still require remaining budget. Live or unknown worker PIDs
prevent launching new workers. Inspect and terminate orphan workers first.
Never clear budget charges to obtain more than the declared cap.
All worker stderr/stdout remain under `logs/`. A nonzero worker exit or timeout
is a nonzero launcher exit; partial output is not a completed scientific result.

The remote 176-start campaign has not been run as part of implementation.
Its timing, full archive coverage and held-out skill must be established with
the commands above before claiming scientific success.
