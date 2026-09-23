# Four-method GPU screen and forecast ensemble

Run four small neural experiments concurrently, one optimizer and one GPU per
method. This is model parallel experimentation; each model uses a batch of one.
It is separate from the older launcher that spreads one model across four GPUs.

| GPU by default | Method | Learned components |
|---|---|---|
| 0 | `full` | Encoder, atmospheric tendencies, decoder |
| 1 | `tendency-only` | Atmospheric tendencies; identity encoder/decoder |
| 2 | `adapters-only` | Encoder/decoder; physical atmospheric evolution |
| 3 | `decoder-only` | Output correction of the physical forecast |

Every method retains physical PBL mixing, uses the same observed-atmosphere/soil
initialization, terrain, seed, normalization data, training order, horizon, and
validation cases. The fourth method is a direct test of whether output correction
alone explains improvements. All runs start at the physical baseline. Different
methods have different active parameter counts and compute costs; this is an
equal-update comparison, not an equal-FLOP comparison.

## Run on the remote server

From the repository root on branch `neural/four-method-screen`, with the Python
environment activated, CUDA-enabled JAX installed, and `MOLA_PATH` pointing to
the MOLA image (or use `--mola /path/to/mola.img`):

```bash
bash scripts/launch_neural_screen_4gpu.sh
```

Defaults: GPUs `0,1,2,3` (or four indices already in `CUDA_VISIBLE_DEVICES`),
25 updates per method, width 32, one residual block, one native forecast interval,
four training chunks, four fixed validation examples, validation at step zero and
every five updates, and a **30-minute wall-time limit for the whole screen**.
The first compile/download/normalization is included in that limit. Set a larger
limit only after inspecting timing. Each method receives one GPU through its own
`CUDA_VISIBLE_DEVICES` and `--devices 1`; the parent does not initialize JAX.

Explicit short-run command:

```bash
bash scripts/launch_neural_screen_4gpu.sh \
  --gpus 0,1,2,3 --steps 25 --minutes 30 \
  --validate-every 5 --validation-examples 4 \
  --output outputs/neural_screen/deadline
```

Inspect the exact four commands without launching workers:

```bash
bash scripts/launch_neural_screen_4gpu.sh --dry-run
```

After useful validation results, resume all four toward 100 total updates:

```bash
bash scripts/launch_neural_screen_4gpu.sh --resume --steps 100 --minutes 60
```

Repeat any custom data/architecture/output flags. The default `--budget-steps 100`
fixes the curriculum endpoint across resumes; changing it or comparison settings
requires a new output directory. All four checkpoints must exist to resume the
screen. If a worker fails before writing its step-zero checkpoint, inspect its
`console.log`, fix the cause, and use a new output directory. Training snapshots
and normalization remain shared through the checksummed, locked source cache.
Do not load a PBL checkpoint into this screen.

## Logs and outputs

The terminal and `tables.log` show every worker output line in a table with GPU,
method, event, step, training loss, validation loss, gain percentage and timing.
Long detail text is shortened for display; each worker's `console.log` preserves
the complete stdout/stderr, including tracebacks. Its existing `run.log`,
`training.jsonl`, and `validation.jsonl` retain structured events and metrics.
These files append on resume.

Whenever all four methods produce the same validation step, another table shows:

```text
| Method        | Physical   | Validation | Internal   | Gain %     |
|---------------|------------|------------|------------|------------|
| full          | ...        | ...        | ...        | ...        |
| tendency-only | ...        | ...        | ...        | ...        |
| adapters-only | ...        | ...        | ...        | ...        |
| decoder-only  | ...        | ...        | ...        | ...        |
| ensemble      | ...        | ...        | -          | ...        |
```

`summary.json` records worker states, exit codes, latest validation reports and
paired ensemble results. `ensemble-step-000025.json` contains that step's scores,
per-case improvements and per-field/layer RMSE. Each method has its own
`checkpoint.pkl`, `best_validation.pkl`, and
`validation-predictions/step-000025.npz`. Forecasts are exported during the existing
validation call, without an additional physical rollout. Failed workers do not
cancel healthy ones; failures and time limits produce a nonzero runner exit code.
The runner interrupts workers at its time limit, then kills any that have not
exited after a 15-second grace period. Completed atomic checkpoints remain usable.

The equal-weight ensemble averages the **four forecast fields**, not their losses.
It rejects differing steps, case times, targets, mass weights, normalization,
terrain or comparison contracts. Weights are fixed at 0.25; no weights are fitted
on validation targets. The ensemble is a forecast product, not a single coupled
GCM state, so it has no internal-state score.

## Interpreting the screen

Read the comparison at a common step, not four asynchronously updated losses.
Compare improvement against the common physical baseline, internal versus decoded
loss, and case/field consistency. At 25 steps, the purpose is fast feedback, not a
convergence claim. A decoder-only win supports neural postprocessing; improved
physical tendencies require evidence beyond decoded error alone. The ensemble
may help, match, or underperform the best member.

The four validation examples and short training budget are for method selection.
Freeze the selected method/checkpoint before independent test-period evaluation.
Real four-GPU throughput, long-horizon stability and useful Mars forecast gains
must be established on the server; local tests do not establish those results.

## Local verification

```bash
python -m pytest package/tests/gcm3d/test_neural_screen.py \
  package/tests/gcm3d/test_neural_hybrid.py \
  package/tests/gcm3d/test_neural_hybrid_resume.py -q
```

The supervisor tests start four real subprocesses with lightweight synthetic
workers to verify isolation, failure handling and time limits. Numerical tests
verify forecast averaging and reject unpaired artifacts. Trainer tests verify
exported forecast scores and exact checkpoint resume; the marked slow test checks
actual coupled gradients and controlled learning. Four-GPU training is not run
by these local tests.
