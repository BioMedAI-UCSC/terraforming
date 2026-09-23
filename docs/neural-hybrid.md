# NeuralGCM-inspired Mars hybrid

For the short four-GPU method comparison with table logs and paired forecast
ensembling, use the [four-method screen](neural-screen.md).

This implementation retains the physical Mars GCM and learns additive atmospheric
**temperature and wind tendencies across the full column**. It addresses the
limited influence and saturation of the original two-multiplier PBL closure.
It is a research-supported architecture, not a pretrained Mars model or a promise
of forecast skill. Existing PBL scripts/checkpoints remain separate.

## Research basis and adaptation

Primary reference: Kochkov et al., **Neural general circulation models for weather
and climate**, Nature (2024), [paper](https://doi.org/10.1038/s41586-024-07744-y),
[full text and appendices](https://arxiv.org/html/2311.07222).
Appendices 3.4–3.5 describe shared column networks, learned encoder/decoder
corrections, and small tendency scaling. The training appendix describes a
progressively longer rollout curriculum and losses in both model and data space.
The public [EncodeProcessDecodeTower implementation](https://github.com/google-research/neuralgcm/blob/9210405b708b37821db104c8cbeb6652c029fe13/neuralgcm/legacy/towers.py)
is a second primary reference, pinned to an inspected commit.

The Aircast-Mars example is a direct atmospheric state forecaster, so its
architecture is not used inside this hybrid.

| Element | Implemented Mars adaptation |
|---|---|
| Column network | Linear encode, five residual blocks of three width-384 dense layers, GELU, linear decode; shared over horizontal locations |
| Learned modules | Separate encoder, physics, decoder networks; zero output weights/biases give identity adapters and zero tendencies initially |
| Inputs | Whole-column T/u/v, vorticity/divergence, pressure, horizontal temperature gradients; surface/soil temperature, frost, dust, solar forcing, location, terrain |
| Learned outputs | T/u/v state increments in adapters; T/u/v per-second residuals in physics; no bounded coefficient sigmoid/tanh |
| Scaling | Frozen training-only feature means/std; 0.01 times training field or finite-difference tendency std for outputs |
| Coupling | Residual held fixed for up to 1800 seconds, then refreshed; existing physical processes still execute at every GCM step |
| Initialization | Observed atmosphere at forecast start; implicit soil conduction driven only by preceding observed surface temperatures |
| Stability | MOLA terrain, existing CO₂ positivity projection, fourth-order spectral diffusion with 0.1-sol shortest-wave timescale |
| Curriculum | 3 native intervals through update 1000, 6 through 2000, 12 through 4000 (~6h, half-sol, one sol) |
| Objective | Area/layer-mass weighted standardized forecast MSE, 0.5 internal-state MSE, 0.1 global bias loss, adapter round-trip error, scaled-output regularization |

Differences from the Earth research model are deliberate: this retains the
existing Mars radiation, convection, PBL, surface, and soil physics; omits moisture,
stochastic fields, learned spatial embeddings, and the paper's filtered spectral
amplitude objective; uses observed sigma-grid targets for its internal-state loss;
and starts with shorter Mars rollouts. It is **not an exact NeuralGCM reproduction**.
The research model's multi-day training/stability results do not transfer automatically.

Learned tendencies do not directly modify pressure, frost, surface temperature,
or soil. They are explicit heat/momentum sources, **not conservative internal
PBL exchanges**. Net temperature mass source and tendency RMS are logged. A learned
atmospheric correction can still indirectly affect physical CO₂ condensation.
Climate drift, energy/work budgets, and altered-composition terraforming scenarios
need separate evaluation; present-day MACDA alone does not establish that skill.
Dust is held at the initial snapshot during each forecast. Native endpoints align
exactly by dividing one native interval into an integer number of timesteps.

## Remote commands (activated mamba environment)

Run from the repository root. These commands use `python` from the active
environment; no `rtk`, `torchrun`, or `mamba run --` wrapper is needed.
The original data dependencies still apply, including HTTP support:

```bash
python -m pip install aiohttp requests
export MOLA_PATH=/absolute/path/to/mola.img
```

A small physical/identity check, using eight held-out cases and a three-interval
forecast, saves frozen normalization and a step-zero checkpoint:

```bash
bash scripts/launch_neural_hybrid_4gpu.sh --preflight-only
```

Resume that checkpoint for synchronized four-GPU training:

```bash
bash scripts/launch_neural_hybrid_4gpu.sh --resume
```

All four GPUs contribute gradients to **one optimizer update**. Input/soil
preparation, normalization, and validation use one device. Compilation occurs
when entering each new horizon. Logs include downloads/cache access, normalization,
physical validation, compilation-stage transitions, update loss/gradient norms,
progress/ETA, checkpoint events, validation scores, internal-state loss,
per-layer T/u/v RMSE, and learned tendency diagnostics. Long phases emit heartbeats.
Validation runs at step zero, every 100 updates, every data epoch, and the final
requested step. Set `--validate-every-steps 50` for more frequent monitoring.

A separate, smaller **real-data optimizer smoke** exercises downloads,
normalization, physical validation, actual coupled updates, and validation:

```bash
bash scripts/launch_neural_hybrid_4gpu.sh \
  --output outputs/neural_hybrid/smoke \
  --width 32 --blocks 1 --chunks 1 --normalization-chunks 1 \
  --curriculum 1:2 --steps 2 --validation-horizon 1 \
  --validation-examples 2 --validate-every-steps 1
```

That smoke model is a separate experiment: do not resume it with production
width/horizon settings. Full training uses the defaults (384, five blocks).
A fresh production run may also skip the separate preflight command: initial
validation always executes before optimization.

Stage chunks without training:

```bash
python scripts/train_neural_hybrid.py \
  --revision 65a0bebd804b9c240752277e83f5737d58c6ee9c \
  --output outputs/neural_hybrid/staging --stage train --chunks 7
```

Evaluate the trained checkpoint (same architecture/data options as training):

```bash
bash scripts/launch_neural_hybrid_4gpu.sh --evaluate validation
bash scripts/launch_neural_hybrid_4gpu.sh --evaluate test
```

Evaluation uses `--validation-horizon` and all windows in the selected split
unless the original run used `--chunks`. Test data never set normalization or
choose checkpoints. Current `--evaluate` loads `checkpoint.pkl`; for selecting
`best_validation.pkl`, copy it into a separate evaluation output directory as
`checkpoint.pkl`, retaining the same training arguments. Do not overwrite your
resumable training checkpoint.

## Ablations and deciding whether training helps

Run each mode with the same seed, terrain, data selection, curriculum, and
validation subset, in distinct directories:

```bash
bash scripts/launch_neural_hybrid_4gpu.sh --ablation tendency-only \
  --output outputs/neural_hybrid/tendency_only
bash scripts/launch_neural_hybrid_4gpu.sh --ablation adapters-only \
  --output outputs/neural_hybrid/adapters_only
```

The default `full` mode learns all three networks. `tendency-only` freezes both
adapters at identity; `adapters-only` has zero learned physics tendencies. Every
mode reports a physical baseline with exactly the same terrain, diffusion, soil
history, initialization time, and forecast endpoints. These ablations determine
whether improvement requires learned dynamics or comes from adapter corrections.
They are implemented experiments, not completed results.

`decoder-only` additionally freezes the encoder, learning only an output
correction of the physical forecast. The screen evaluates all four modes with
identical settings and can export validation predictions for a fixed equal-weight
ensemble; see the linked screen instructions.

Judge fixed held-out forecast loss and per-layer errors, not loss across changing
training batches or curriculum stages. Compare internal-state loss with decoded
loss to detect adapter compensation. Eight cases provide a monitoring check, not
statistical evidence; use broader validation and multiple seeds before reporting
skill. The old PBL validation scores are **not directly comparable**, because
initialization, terrain, diffusion, horizon, and normalization changed.

Checkpoints atomically preserve parameters, Adam moments/count, frozen statistics,
data cursor, shuffled chunk order, NumPy RNG state, architecture, curriculum,
revision, terrain checksum, and training contract. Dropout is not used. A contract
mismatch (including old PBL checkpoints) is rejected. Only load trusted checkpoint
files, since the serialization uses pickle.

## Local verification

```bash
python -m pytest package/tests/gcm3d/test_neural_hybrid.py -q
python -m pytest package/tests/gcm3d/test_neural_hybrid_resume.py -q
python -m pytest package/tests/gcm3d/test_neural_parallel.py -q
```

The coupled derivative/learning test uses a short physical rollout and a known
synthetic teacher tendency. It compares encoder, physics, and decoder derivatives
with centered finite differences, then requires learning the physics correction
alone to reduce the objective by more than half in eight updates.

The trainer resume test uses inexpensive deterministic surrogate physics with the
real training loop, Adam updates, and atomic checkpoint files. It compares an
uninterrupted run against preflight and repeated resumes, crossing curriculum and
shuffled-epoch boundaries. Parameters, optimizer moments, frozen statistics, data
cursor, chunk order, RNG state, and loss/gradient histories must match exactly.
It also checks contract rejection and rejects nonfinite internal validation loss
even when the decoded loss is finite. The parallel test checks gradient averaging
and optimizer restoration on four virtual CPU devices.

These checks verify implementation behavior; they do not demonstrate real-data
Mars generalization, production-horizon stability, or four-GPU throughput.

Local verification on 2026-09-22: all seven hybrid/resume/parallel tests passed,
including the marked slow coupled test. Ten adjacent PBL, validation, logging,
and diagnosis regression tests also passed (the separate slow PBL test was
deselected). Launcher shell syntax and trainer CLI startup passed.
