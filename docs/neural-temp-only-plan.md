# Temperature-only neural postprocessing: implementation plan

Status: implemented; see [launch and acceptance runbook](neural-temp-only-runbook.md).
Local verification is separate from the still-required bounded remote experiment
and its held-out scientific criteria.
Branch: `neural-temp-only`, created directly from `final-phase-2`.

## Objective and scope

Improve a frozen physical Mars GCM's lowest-model-layer temperature forecast
against MACDA using a small neural residual corrector. Target one fixed lead:
one native MACDA interval, exactly one twelfth of a Martian sol (approximately
two Mars hours). The target is air temperature at the model's lowest sigma
midpoint, approximately sigma = 0.9583 for 12 uniform layers. It is not surface
temperature, measured 2-m temperature, or a fixed geometric altitude.

The forecast is `T_corrected = T_physical + neural_residual(features)`.
Only the delivered temperature field changes. Physical PBL, radiation, winds,
pressure, surface exchange and CO2 processes continue to determine the physical
forecast. The correction is not fed back into the simulated state. The physical
solver is frozen and forecast generation is amortized through a disk cache.

The intended claim is improved reanalysis-referenced temperature prediction
from a physical forecast and a small neural corrector. This experiment does not
establish improved mixing, climate stability, or the benefit of differentiating
through the GCM. Coupled correction, online adaptation after new observations,
multiple forecast leads and learned initial conditions are outside this scope.

## Evidence and success criteria

The earlier four-case screen on `neural/four-method-screen` found a mass-weighted
lowest-layer physical RMSE of about 3.94 K and mean bias about -0.50 K. An oracle
constant bias correction would improve RMSE by only about 0.8%; the existing
decoder-only checkpoint improves lowest-layer RMSE by about 0.44%. Those four
cases are development evidence, not a new independent test. Their absolute
scores must not be reused as this branch's baseline: the physical configuration
and initialization must be regenerated and recorded on this branch.

Research supports conditional correction, but does not guarantee a Mars gain.
Rasp and Lerch report roughly 29–33% CRPS improvement over raw Earth forecasts,
and only approximately 2.5–3.5% over their strongest boosted statistical control.
CRPS improvements are not deterministic RMSE improvements. Representative data,
location information and auxiliary physical predictors matter more than adding
network depth. Previous training used the first four adjacent source chunks;
this implementation must not repeat that coverage limitation.

Declare these criteria before training:

* Primary metric: area-weighted RMSE in kelvin, with equal forecast-start weight.
  Compute each start's area-weighted MSE, average across starts, then take the
  square root. Do not average start RMSEs or flatten grid cells as independent
  weather examples. The earlier 3.94 K diagnostic used a different, mass-weighted
  aggregation; report mass-weighted RMSE secondarily for context.
* Substantial gain: at least 10% relative primary RMSE reduction versus the
  paired physical baseline on the fixed test manifest. This is a target, not a
  prediction; it requires eliminating at least 19% of baseline MSE.
* Neural value: lower test RMSE than the fitted linear control, with a paired
  temporal-block bootstrap interval for MSE improvement excluding zero.
* Physical forecast value: improvement over the matched neural model without
  physical forecast inputs. Otherwise do not claim the GCM adds predictive value.
* Report season, latitude band, and day/night metrics even when they regress.
  Uncertainty resampling uses temporal blocks, never individual grid cells.
  One small dataset and one seed give limited evidence; label conclusions so.

Failure to meet a criterion must be reported. Do not change the target layer,
lead, geographic mask, metric or test cases after observing results.

## 1. Define data and forecast contracts

Reuse the base branch's coordinate system, physical equations, positivity
wrapper, MOLA terrain handling and spatial regridding utilities. Its
`scripts/stage_arco_macda.py` stages daily means; add a separate native-cadence
loader rather than changing the meaning of existing daily products. The old
neural trainers are absent from this branch. Their observed-atmosphere and
past-only soil initialization may be adapted selectively after review; do not
merge their full branch or unrelated physics edits.

Pin the source revision to
`65a0bebd804b9c240752277e83f5737d58c6ee9c`, and record the MACDA and ARCO citations.
Read actual year/time coordinates when selecting windows: the archive contains
out-of-order blocks, so do not binary-search a presumed globally monotonic time
axis. Validate each proposed history/start/target interval for finite values,
units, native cadence and split membership. Exclude and log invalid intervals;
never interpolate across gaps or silently substitute a different target.

Preserve the existing chronological split definition:

| Split | Mars years | Use |
|---|---|---|
| Train | 24, 25, 26, 27, 29, 30, 31 | Features, normalization, all fitted parameters |
| Validation | 32, 33 | Early stopping and frozen model selection |
| Test | 34, 35 | One final evaluation of frozen models |
| Excluded stress set | 28 | No tuning or headline score in this deadline study |

Initial manifest size: 112 training starts, 32 validation starts and 32 test
starts. For each training year select four starts in each of four seasonal
quadrants; for each validation/test year select four starts in each quadrant.
Choose dates using metadata and a fixed seed, not errors. Require disjoint
history-to-target intervals and at least two sols between starts. Balance dates
within a quadrant rather than selecting its first four valid records. Every
global snapshot already spans local times across longitudes; verify geographic
day/night coverage rather than treating a UTC timestamp as one global local time.

Freeze manifests before fitting. Log requested and achieved coverage, exclusion
reasons and the actual source indices/times. If valid data cannot meet coverage,
stop and report the deficit; do not silently shrink the experiment. Use source
file provenance where available to characterize observing-system changes. A
global attribute carried through concatenated ARCO data is not sufficient proof
of the assimilation regime for every window. Mark unknown provenance explicitly.

The forecast contract includes code revision, dataset revision, manifest hash,
T21/L12 grid, sigma centers, physical forcing/options, timestep, precision,
terrain hash, source variables, initialization, feature schema and lead time.

## 2. Generate reusable physical forecast pairs

For each start:

1. Initialize atmospheric T/u/v and pressure from MACDA using the existing
   spectral projection conventions. Initialize surface temperature and frost
   from the start only.
2. Initialize the regolith from the preceding one-sol surface-temperature
   history using physical conduction with prescribed historical boundary
   temperatures. Specify and test the initial deep-soil assumption; use exactly
   the same procedure for every model. No future surface observations enter it.
3. Advance the physical GCM one native interval. Use the base physical processes
   with PBL, regolith, stability exchange and convective adjustment enabled;
   freeze the exact configuration before producing any training labels. Use
   MOLA terrain, float64 and an interval-dividing timestep no larger than 300 s.
4. Keep start dust forcing fixed over the forecast, unless an explicitly
   available-at-initialization forcing forecast is introduced in the contract.
   Solar/orbital evolution is deterministic. Do not substitute observed future
   dust or future surface temperature as predictors.
5. Cache start features, physical forecast features, target temperature, area
   weights, metadata and the physical error. Targets live separately from
   inference features. Record pressure separately for secondary mass metrics.

Physical forecasts are independent jobs: shard start IDs across four GPUs with
one masked worker per GPU. Check actual device placement. Cache files are
written atomically with checksums; completed matching entries are reusable.
Contract mismatch must refuse cache reuse. A crash resumes missing start IDs,
not a repeated full simulation campaign.

Do not consume the old neural screen's forecast arrays as new training or test
data. They remain development diagnostics with a different contract.

## 3. Feature schema and neural model

Use one shared MLP: `features -> Dense(32) -> GELU -> Dense(32) -> GELU -> Dense(1)`.
The output is a correction in kelvin after multiplying by a frozen training-only
residual RMS. Initialize the last layer to zero, so the initial forecast exactly
reproduces the physical baseline. This blocks hidden-layer gradients only on the
first update; it is not a justification for a random initial temperature change.

Initial feature schema, all continuous and finite:

* Start lowest-layer T, start surface T, start next-layer T, start u/v, pressure,
  and column dust opacity.
* Forecast lowest-layer T and its change from start; forecast surface T and
  surface-minus-lowest-air T; next-layer-minus-lowest-layer forecast T; forecast
  lowest-layer u/v and surface pressure.
* Forecast-time insolation, sine/cosine of local solar time, sine/cosine of solar
  longitude, sine of latitude, sine/cosine of longitude, and terrain height.

Local time and orbital-angle conventions must use the physical solver's
conventions. Do not approximate Mars' seasonal progression by a uniform angle
if the solver uses eccentric orbital evolution. No target-derived quantities,
future reanalysis predictors, case IDs or dataset-year IDs enter the network.
No learned per-cell embeddings are needed for the initial implementation.

Freeze feature means/scales and residual RMS using training starts only, with
positive scale floors. Preserve feature order and units in the checkpoint.
Missing required predictors fail explicitly rather than being replaced with
validation-derived values.

## 4. Fit cheap, meaningful controls on the same cache

| Model | Definition | Purpose |
|---|---|---|
| Physical | Uncorrected cached GCM forecast | Main reference |
| Persistence | Start lowest-layer temperature | Basic forecasting control |
| Constant residual | Training-weighted mean physical error | Global bias control |
| Linear residual | Ridge regression on the same standardized predictors | Tests value beyond conditional linear correction |
| Neural residual | Shared 32/32 MLP plus physical forecast | Main candidate |
| Neural without GCM forecast | Same hidden widths; predicts change from start T using start/static/deterministic features only | Tests added value of the GCM forecast |

The last control removes both forecast feature channels and the physical
forecast additive term. It predicts `T_start + NN(start, static, solar features)`;
retaining `T_physical` would invalidate the ablation. Normalize its residual from
its own training targets. All controls use identical starts, target field,
metric, weights and available-at-initialization information.

For neural models use area-weighted squared temperature error, Adam at 0.001,
gradient-norm clipping at 1 and weight decay 0.0001. Use a fixed seed of zero,
at most 100 epochs, validation once per epoch, and patience of 10 epochs.
Retain the best validation checkpoint including the zero-correction checkpoint.
These are fixed initial settings, not a promised optimum or a search grid.
For the standardized ridge objective use mean weighted squared error plus
0.001 times coefficient squared norm; leave its intercept unpenalized.

Shuffle forecast starts each epoch. A batch contains columns from several
different starts: sample eight starts uniformly and 128 columns per start by
area probability. This estimates equal-start, area-weighted loss; do not apply
area weights a second time to already area-proportional samples. Full-grid
validation always uses explicit quadrature weights. Preserve sampler RNG state
for deterministic resume.

Training the two small neural models can use one GPU or CPU; do not reserve four
GPUs for these regressions. No GCM rollout occurs inside a training update.

## 5. Evaluation, artifacts and table logs

Select each model using validation only, then freeze all models before test.
Evaluate all controls once on the same test starts. Do not adapt parameters
during test or choose a different model based on test performance.

Write per-start predictions and paired squared-error differences, aggregate
RMSE/MAE/bias, secondary mass-weighted metrics, and prespecified breakdowns:
four seasonal quadrants, latitude bands [-90,-60), [-60,-30), [-30,30),
[30,60), [60,90], and day/night using solar zenith cosine. Record sample weights
and counts for each group. Bootstrap temporal blocks defined in the manifest;
publish their definitions, seed and 95% paired intervals. Do not report millions
of grid cells as independent samples.

Add error maps, predicted-versus-reference temperature, residual-versus-local-time
plots and paired per-block skill plots. Any oracle correction calculated from
validation labels is labeled diagnostic and excluded from score comparisons.

Output layout: `outputs/neural_temp/<run_id>/` containing `contract.json`,
`manifest.json`, `normalization.json`, cache index/checksums, fitted baseline
parameters, `checkpoint.pkl`/`best_validation.pkl` for each neural model,
`training.jsonl`, validation/test reports, prediction arrays, plots and a concise
Markdown report. Checkpoints include architecture, parameters, optimizer state,
epoch/update count, sampler state, feature schema and contract hashes. Refuse
resume when any scientific contract changes.

Print readable tables during caching and training: stage, GPU, completed/total
starts, epoch, train RMSE, validation RMSE, physical RMSE, relative gain, elapsed
time and aggregate GPU-hours. Preserve structured JSONL events and raw worker
logs. A successful command exit requires all requested artifacts and complete
coverage; worker errors and timeouts return failure while preserving progress.

## 6. Compute cap and decision gates

The conservative budget is four **aggregate GPU-hours**, including compilation
and time GPUs are reserved during data preparation. Four GPUs for one hour
consume the entire budget. CPU-only metadata work does not consume this budget.

| Stage | Aggregate GPU-hour cap |
|---|---:|
| Training/validation forecast caching | 2.25 |
| Small-model training | 0.25 |
| Test forecast caching and evaluation | 1.00 |
| Contingency | 0.50 |

Time compilation and the first few requested forecast jobs, retaining their
cache entries. Project the full manifest cost before continuing. If the workload
cannot fit, report the estimate and propose a revised scope before spending the
evaluation reserve. Do not silently reduce coverage or skip controls.

After validation, proceed to final test only if the neural candidate meets the
10% RMSE target versus physical and beats the linear control on validation.
Otherwise report that substantial improvement has not been demonstrated and
stop; do not start an architecture sweep. Passing this gate does not guarantee
test success. Do not retrospectively claim a test result if the gate stops it.

## 7. Implementation sequence and verification

| Deliverable | Proposed files | Required verification |
|---|---|---|
| Native manifests and contracts | `scripts/neural_temp_data.py` | Gaps/out-of-order blocks, year boundaries, nonoverlapping splits, deterministic selection |
| Frozen forecast cache | `scripts/cache_neural_temp.py` | Observed initialization, no future predictors, physical identity, crash/resume and contract rejection |
| Feature builder and residual MLP | `scripts/neural_temp_model.py` | Units/order, training-only statistics, zero output identity, finite gradients |
| Regressions and controls | `scripts/train_neural_temp.py` | Weighted analytic baseline checks, meaningful synthetic learning, resume equivalence, removal of all GCM inputs in ablation |
| Evaluation/reporting | `scripts/evaluate_neural_temp.py` | Hand-computed weighted metrics, paired cases, temporal bootstrap units, no test fitting |
| Bounded GPU runner | `scripts/run_neural_temp.py` | Unique device routing, atomic progress, timeout/worker-failure propagation, table/JSONL logs |

Place focused tests under `package/tests/gcm3d/`. Reuse physical unit checks from
the base branch; add only tests needed for data leakage, scientific metrics,
model behavior and restart reliability. A synthetic fit demonstrates optimizer
correctness, not Mars generalization. First verify locally on synthetic arrays
and cached metadata; run the bounded remote workflow only after implementation
and its concrete launch command are reviewable.

Implementation is complete when the contract, manifest, cache, all controls,
resume path, bounded launcher and reporting work together and pass their checks.
Scientific success additionally requires the declared held-out criteria; working
software or reduced training loss alone does not satisfy them.

## References

1. Rasp and Lerch, *Neural Networks for Postprocessing Ensemble Weather Forecasts*,
   Monthly Weather Review (2018), DOI 10.1175/MWR-D-18-0187.1.
   [Paper, especially Table 2 and data/evaluation sections](https://arxiv.org/html/1805.09091).
2. Gronquist et al., *Deep learning for post-processing ensemble weather forecasts*,
   [DOI 10.1098/rsta.2020.0092](https://doi.org/10.1098/rsta.2020.0092).
   Its reported skill gains concern probabilistic ensembles, not Mars RMSE.
3. Valeanu et al., MACDA v2.0 (2026),
   [CEDA dataset and observing-system description](https://catalogue.ceda.ac.uk/uuid/cd037a9ea387438fabf4d674dbe53088/),
   DOI 10.5285/cd037a9ea387438fabf4d674dbe53088.
4. Bhattacharya, ARCO-MACDA (2026),
   [dataset card](https://huggingface.co/datasets/ananyo01/ARCO-MACDA),
   DOI 10.57967/hf/8771. The implementation pins a revision rather than `main`.
