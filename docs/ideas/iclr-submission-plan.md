# ICLR submission plan: a differentiable neural GCM for Mars

**Living document — last updated 2026-09-05**  
**Internal submission deadline: 2026-09-26 (21 calendar days)**  
**Paper freeze: 2026-09-24; upload and compliance buffer: 2026-09-25–26**

> Verify the official ICLR year, time zone, abstract-registration date, and paper
> deadline in the submission portal today. This plan treats September 26 as the
> user-specified hard deadline, not as a verified conference date.

## North-star goal

Build and evaluate a **neural-GCM-inspired, end-to-end differentiable simulator
for present-day Mars**: a Dinosaur/JAX primitive-equation dynamical core coupled
to differentiable, conservation-aware Mars physics, with a constrained learned
residual or gradient-based calibration experiment. The paper must show what
differentiability buys, not merely that the simulator was rewritten in JAX.

The submission is not a claim of a complete Mars Planetary Climate Model (PCM),
nor a terraforming-results paper. The water cycle, interactive dust lifting,
cloud microphysics, photochemistry, and intervention scenarios are out of scope
unless the minimum paper is already complete.

## Proposed paper claim

**Working title:** *A Differentiable General Circulation Model for Mars*

**One-sentence claim:** We introduce a modular differentiable Mars GCM that
preserves explicit physical structure and budgets, supports gradients through
multi-step atmospheric rollouts, and improves a deterministic baseline through
constrained learning or gradient calibration while remaining stable under
seasonal and out-of-distribution evaluation.

The paper needs all four pillars:

1. **Simulator:** 3-D primitive-equation dynamics plus a credible dry-Mars
   physics package, end-to-end differentiable and restartable.
2. **Verification:** standard dycore tests, isolated physics tests, conservation,
   gradient correctness, runtime/memory, and rollout stability.
3. **Mars evaluation:** observations/reanalysis as the primary empirical target;
   LMD PCM/Mars Climate Database (MCD) and NASA Ames MGCM as model references.
4. **Learning result:** at least one reproducible result where gradients or a
   constrained residual model improve held-out performance over the deterministic
   model and simple tuning baselines.

If pillar 4 is not obtained by September 17, reposition the paper as a
differentiable-simulator/benchmark paper only if the venue fit and results are
strong enough; do not imply a neural model that was not trained.

## Naming and evidence rules

- Use **LMD Mars PCM** for the Laboratoire de Météorologie Dynamique model and
  **MCD v6.1** for its database product.
- Use **NASA Ames Mars GCM (Ames MGCM)** for the second model reference.
- “MSTCM,” “MSDCM,” and “PCM LMB” are not used until their intended expansions
  are confirmed. They are likely transcription variants of Mars GCM / LMD PCM.
- Confirm the exact title, DOI, license, variables, spatial/temporal coverage,
  and download mechanism for the user-mentioned **ArcoMars** resource before it
  enters the methods section. Do not build the schedule around an unverified name.
- MCD and Ames are simulated climates, not observational truth. A model trained
  on one is evaluated on held-out periods from that source and independently on
  spacecraft observations/reanalysis; agreement with MCD alone is not validation.
- Every table identifies run length, spin-up, resolution, dust scenario, Mars
  year/season, local time, vertical coordinate, and whether the value is a
  transient diagnostic or an equilibrated climatology.

## Current evidence base (repository audit, 2026-09-05)

### Implemented and tested

- Dinosaur spectral primitive-equation core, JAX rollouts, replaceable structured
  physics tendencies, differentiability tests, and T21/T42 conservation tests.
- MOLA topography; independent TES bolometric albedo and thermal inertia staging.
- Kepler-consistent seasonal/diurnal forcing; surface-atmosphere energy exchange;
  prognostic surface temperature and four-layer regolith.
- Surface drag, Richardson-dependent exchange, implicit conservative PBL momentum
  mixing, heat/tracer diffusion, and differentiable dry convective adjustment.
- CO2 condensation/sublimation with mass and latent-energy coupling.
- Differentiable two-stream radiation and staged NASA Ames 12-band correlated-k
  CO2 tables; prescribed radiatively active dust operators.
- Exact restart machinery, averaging windows, NetCDF/maps, a restartable 668-sol
  driver, and an MCD v6.1 comparison harness.
- At commit `8314609cac7fcd91c6a38dc8440392493c7eccc2`, the focused GCM
  non-slow suite passes: **125 passed, 3 skipped, 2 deselected in 74.43 s**.
  The complete package non-slow suite also passes: **290 passed, 3 skipped,
  5 deselected in 102.39 s**. These are local CPU results from Python 3.12.3;
  the five deselected tests are two GCM climate/convergence benchmarks and three
  long 0-D intervention/pressure-budget tests. The three skips are stale parity
  cases for an older three-element JAX seasonal state after the Torch reference
  moved to a four-element two-cap state; they are known technical debt, not
  missing dependencies. These exclusions still require resolution or an explicit
  submission rationale.
- The two slow GCM benchmarks also pass independently after making their required
  JAX float64 precision explicit: **5/5 tests in the complete benchmark module
  pass** (33.48 s on the warm local compilation cache). This removes hidden test
  order dependence from the roundoff-level conservation claim.

### Existing quantitative evidence (diagnostic only)

The current checked-in MCD comparisons are 3.5-sol transients at Ls=0 and cannot
support climatology claims. The most favorable drag-only matched-height result has
surface-temperature bias **+7.56 K**, RMSE **11.35 K**, and spatial correlation
**0.901**; wind-speed bias **+0.95 m/s**, RMSE **6.90 m/s**, and correlation
**0.184**. Adding TES surface structure/regolith/Richardson/PBL/convection raises
temperature correlation to **0.920** but worsens RMSE to **13.30 K**, while the
wind RMSE rises to **18.97 m/s**. These numbers diagnose spin-up/PBL problems;
they are not paper headline results.

### Open scientific blockers

- The full deterministic coupled model has not completed a documented one-year
  spin-up plus one-year evaluation run.
- Correlated-k radiation still needs a published Mars column benchmark.
- Regolith needs sinusoidal amplitude/phase and timestep-convergence acceptance.
- PBL needs stable nocturnal and daytime LES/single-column benchmarks.
- CO2 condensation needs Viking seasonal-pressure validation.
- Prescribed dust needs a versioned seasonal scenario and comparison protocol.
- No learned residual module, training pipeline, dataset contract, or held-out
  learned-model result exists yet.
- No matched Ames MGCM reference experiment is recorded yet.

## Minimum publishable experiment

Freeze a single canonical configuration by September 10:

- **Production grid:** T21 with 12 sigma levels and a **300 s timestep** for the
  deterministic baseline and initial calibration. A 600 s step is rejected for
  this configuration because it becomes non-finite within one sol. Use one T42
  evaluation run for resolution sensitivity if compute permits; do not assume
  its throughput from T21.
- **Physics:** correlated-k CO2 radiation, prescribed climatological dust, TES
  surface fields and regolith, PBL/drag, dry adjustment, and CO2 surface cycle.
- **Learning task:** prefer low-dimensional gradient calibration first (dust
  opacity scale, surface exchange/roughness, thermal-inertia scale, and at most
  one radiation multiplier). In parallel only if capacity exists, implement a
  small column-local residual network that predicts bounded temperature and
  momentum tendencies with zero dry-mass tendency and explicit energy diagnostics.
- **Training target:** one versioned reanalysis or combined observational product
  after a 48-hour data audit. MCD may bootstrap/pretrain but cannot be the only
  test target.
- **Primary evaluation:** held-out seasons/local times/vertical levels from
  observations or reanalysis.
- **Independent model comparisons:** matched LMD/MCD and Ames MGCM climatologies.
- **OOD evaluation:** a held-out dust scenario or Mars year, plus longer rollout
  than used during training.

### Required baselines

1. deterministic model with nominal parameters;
2. deterministic model with conventional derivative-free/random-search tuning;
3. gradient-calibrated deterministic model;
4. learned residual model, only if it clears the September 17 gate;
5. persistence or climatology where appropriate for forecast-style metrics.

### Predeclared metrics

- Area-weighted RMSE, mean bias, MAE, and anomaly/spatial correlation for surface
  pressure, surface temperature, atmospheric temperature, and horizontal winds.
- Zonal-mean latitude-pressure temperature and wind error at four seasonal bins.
- Diurnal surface-temperature amplitude and phase at representative sites.
- Viking Lander pressure-cycle amplitude/phase.
- Global dry mass, atmosphere-plus-cap CO2 mass, TOA/surface energy residual, and
  axial-angular-momentum drift.
- Rollout failure rate, longest stable rollout, gradient norm/finite-gradient rate,
  wall time, accelerator memory, and scaling with resolution.
- Spectral kinetic-energy comparison if it can be produced consistently; otherwise
  list it as future work rather than adding an unreviewed metric late.

Report confidence intervals across seeds for learning results. Do not select
hyperparameters on the held-out test season or Ames comparison.

## Dataset decision: finish in 48 hours

Create a dataset card for each candidate with license, citation, URL, variables,
Mars years, cadence, coordinates, missingness, uncertainty, volume, and intended
split. Audit these candidates: the verified ArcoMars resource, EMARS, OpenMARS,
MACDA, direct MCS temperature, TES/THEMIS temperature and surface products,
Viking pressure, MCD v6.1, and Ames MGCM outputs.

Use this role separation:

| Role | Preferred source | Rule |
|---|---|---|
| Boundary/static data | MOLA + TES surface products | Never take boundary fields from the evaluation target |
| Training/calibration | One accessible reanalysis/combined product | Version preprocessing and split by season/Mars year |
| Empirical test | Withheld spacecraft/reanalysis periods | Never used for parameter or checkpoint selection |
| PCM reference | LMD PCM through MCD v6.1 | Matched scenario, altitude, local time, season, and dust |
| Second-model reference | NASA Ames MGCM | Same initial/boundary configuration as far as possible |

**Data gate, September 7:** choose the source that is downloadable, legally usable,
documented, and convertible to the canonical schema by September 9. If no
reanalysis clears the gate, use direct MCS/TES/Viking observations for calibration
and reserve MCD/Ames strictly as secondary comparisons. Do not spend the final two
weeks negotiating access.

Canonical sample keys: `(source, Mars year, Ls, local solar time, latitude,
longitude, vertical coordinate)`. Store source-native values, masks, uncertainty,
conversion metadata, and provenance checksum before regridding.

## Benchmark protocol

### A. Unit and column verification

- Gradient check selected radiation, regolith, PBL, and CO2-cycle parameters
  against centered finite differences in float64.
- Complete the Forget et al.-style radiation column, regolith periodic-forcing,
  PBL stable-night/daytime case, and CO2 frost/pressure tests.
- Publish pass thresholds before model tuning.

### B. Dynamical-core verification

- Retain the existing resting atmosphere, tracer advection, balanced jet, and
  Held–Suarez suite.
- Produce a reproducible table at T21/T42 and at least two timesteps, including
  mass, energy, and AAM drift and runtime.

### C. Present-Mars climate

- Spin up for at least one Mars year and evaluate a subsequent year. Demonstrate
  equilibration with annual-mean energy imbalance, soil-temperature drift, and
  repeatability of pressure/temperature seasonal cycles.
- Evaluate four seasonal bins centered near Ls 0, 90, 180, and 270 and matched
  local times/vertical levels.
- Use exactly the same postprocessing and masks for deterministic, calibrated,
  learned, MCD, and Ames products.

### D. Matched PCM/MGCM experiments

- Pin dust scenario, topography, solar forcing, surface pressure, surface fields,
  resolution/regridding, sampling times, and comparison height.
- For MCD, record every query and cached response checksum.
- For Ames, first make a minimal standard present-Mars run; save namelists, source
  version, build environment, restart/spin-up length, and diagnostics. Use Ames to
  localize disagreements, not to claim ground truth.

### E. Learning/gradient experiment

- Train on short rollouts with a curriculum to longer windows.
- Bound parameter ranges and residual tendencies; project prohibited mass changes;
  diagnose all energy and momentum introduced by the learned term.
- Select checkpoints on validation climate error plus stability/budget penalties.
- Test at longer horizons and a held-out season/dust case without fine-tuning.

## Critical path to September 26

| Date | Deliverable and exit criterion |
|---|---|
| **Sep 5** | Freeze north-star claim, owner list, compute inventory, paper repository, official deadline check, and this plan. |
| **Sep 6–7** | Dataset audit and one-page related-work matrix; select training and empirical-test sources. Data download and license must be confirmed. |
| **Sep 6–9** | Run full tests; repair only paper-blocking regressions. Finish four isolated physics benchmarks and gradient checks. |
| **Sep 8–10** | Freeze canonical configuration, schema, train/validation/test split, metrics, and experiment IDs. Generate a 10-sol end-to-end smoke result. |
| **Sep 10–12** | Complete canonical T21 spin-up infrastructure and launch deterministic year(s). Launch matched Ames standard run. Draft methods and benchmark sections. |
| **Sep 11–14** | Implement low-dimensional differentiable calibration and derivative-free baseline. Demonstrate loss decrease on a tiny overfit case, then launch real runs. |
| **Sep 13–16** | Implement/train constrained residual only if calibration and data pipelines are stable. Produce first held-out comparison and long-rollout stability plot. |
| **Sep 17 — hard gate** | Results review. A paper path exists only if the full pipeline is reproducible and at least one differentiable method beats the nominal baseline on held-out data without unacceptable drift. Freeze model scope. |
| **Sep 18–20** | Finish three seeds/ablations, MCD comparison, Ames comparison, T42 sensitivity if feasible. Freeze main tables by Sep 20 night. |
| **Sep 18–21** | Write complete draft, figures, limitations, impact statement, reproducibility appendix, and related work. No placeholder claims after Sep 21. |
| **Sep 22** | Internal scientific review: claim/evidence audit, leakage check, units, budgets, captions, and citations. |
| **Sep 23** | External cold read and reproducibility run from a clean environment. Fix only correctness and clarity issues. |
| **Sep 24** | Final PDF freeze; anonymization, format, page limit, supplements, artifact links, and author metadata checked. |
| **Sep 25** | Upload full submission and inspect rendered PDF/supplement. |
| **Sep 26** | Buffer for portal or compliance problems; no new experiments. |

## Parallel workstreams and ownership slots

Assign one named owner and one backup today for each stream:

| Stream | Outputs | Owner | Backup |
|---|---|---|---|
| Physics/verification | Four missing column tests; budget table | TBD | TBD |
| Data | Dataset cards; canonical Zarr/NetCDF; splits | TBD | TBD |
| Climate runs | Spin-up, canonical checkpoints, MCD/Ames exports | TBD | TBD |
| Learning | Calibration, baselines, residual model, seeds | TBD | TBD |
| Evaluation | Unified metrics, plots, uncertainty, leakage audit | TBD | TBD |
| Paper | LaTeX, bibliography, figures, daily integration | TBD | TBD |

Daily cadence: 15-minute blocker meeting, immutable experiment registry, results
ingested into the paper that day, and an evening go/no-go update. A run without a
config, commit, seed, environment, and artifact path is not a paper result.

## Experiment registry

Add one row before launching any material run.

| ID | Commit | Config/data version | Seed | Compute | Status | Result/artifact | Paper use |
|---|---|---|---:|---|---|---|---|
| DIAG-MCD-LS0-P1 | existing | 3.5-sol transient, MCD v6.1 | — | local | complete, diagnostic only | `outputs/gcm3d_maps/current-p1-drag_mcd/benchmark.json` | motivation/error analysis only |
| SMOKE-T21L12-DT600 | working tree | full physics, TES, T21/12, 600 s | — | local CPU | rejected: non-finite in first sol | separate failed `outputs/iclr_smoke_t21_l12_dt600/` attempt | stability boundary only |
| SMOKE-T21L12-DT300 | working tree | full physics, TES, T21/12, 300 s | — | local CPU | complete: 10 sols/2,959 steps | `outputs/iclr_smoke_t21_l12_dt300/` | throughput and stability planning |
| SMOKE-T42L12-DT450 | working tree | full physics, TES, T42/12, 450 s | — | local CPU | complete: 1 sol/197 steps | `outputs/iclr_smoke_t42_l12_dt450/` | resolution-cost planning only |
| CANON-DET-Y1 | TBD | frozen Sep 10 | 0 | TBD | not started | TBD | deterministic baseline |
| CAL-GRAD | TBD | frozen Sep 10 | 0–2 | TBD | not started | TBD | primary differentiability result |
| CAL-DFO | TBD | frozen Sep 10 | 0–2 | TBD | not started | TBD | tuning baseline |
| RESIDUAL | TBD | frozen Sep 10 | 0–2 | TBD | gated Sep 17 | TBD | neural result if successful |
| AMES-MATCH | TBD | standard present Mars | — | TBD | not started | TBD | second-model comparison |

## Paper outline and figure contract

1. **Introduction:** why differentiable planetary climate models; precise claims.
2. **Related work:** Mars PCM/MGCMs, Mars reanalyses/data fusion, NeuralGCM/JCM,
   differentiable simulators and hybrid physics-ML.
3. **Model:** Dinosaur dycore, Mars physics, structured tendency interface,
   constraints, differentiation and training/calibration.
4. **Data and protocol:** provenance, splits, observations versus model references,
   matched comparisons, metrics.
5. **Verification:** dycore, column physics, gradients, budgets, runtime.
6. **Results:** deterministic climatology, calibration/residual, held-out/OOD,
   MCD and Ames comparisons, ablations.
7. **Limitations and broader impact:** model incompleteness, model-reference bias,
   reanalysis dependence, compute, and invalidity for terraforming regimes.

Minimum main-paper figures/tables:

- architecture diagram showing dycore, explicit physics, learned residual, and
  conservation projection;
- verification/budget/gradient table;
- seasonal latitude-pressure or global map comparison against empirical target,
  MCD, and Ames;
- learning curves plus long-horizon rollout stability;
- primary held-out metric table with nominal, derivative-free, gradient-calibrated,
  and residual variants;
- ablation table for radiation, dust, surface/regolith, PBL, and learned term.

## Scope cuts and fallback ladder

Cut in this order: T85, water/clouds, interactive dust, CO2 clouds, multiple neural
architectures, many resolutions, terraforming experiments, then the residual
network. Never cut empirical evaluation, conservation reporting, matched sampling,
the deterministic baseline, or reproducibility metadata.

- **Plan A:** constrained residual + gradient calibration, empirical test, MCD,
  and Ames.
- **Plan B:** gradient calibration as the ML contribution, empirical test, MCD,
  and Ames; residual becomes future work.
- **Plan C:** simulator/benchmark paper with strong validation only. Submit only
  after a frank venue-fit review; do not manufacture a neural claim.
- **No-submit condition:** no equilibrated deterministic baseline, no empirical
  evaluation, irreproducible results, material budget failure, or evidence of
  train/test leakage by September 22.

## Immediate checklist (next 24 hours)

- [ ] Verify conference dates, requirements, page limit, anonymization, and compute
  disclosure; register abstract/title if required.
- [ ] Name owners/backups and inventory available CPU/GPU hours and storage.
- [x] Run the full non-slow and focused GCM suites from the current commit. Both
  pass locally; archive a machine-readable report in the experiment pipeline next.
- [ ] Resolve the ArcoMars citation/name and complete the candidate dataset cards.
- [ ] Freeze the canonical variables, coordinates, metrics, and split proposal.
- [ ] Estimate wall time/storage for T21 one-year and T42 sensitivity runs from a
- [x] Measure the T21/12 one-sol and restart-to-ten-sol runs. The steady estimate
  is recorded below; T42 still needs its own measurement and compute reservation.
- [ ] Create the LaTeX paper skeleton and bibliography; draft model and limitations
  directly from the verified repository documentation.
- [ ] Open issues for the four missing physics benchmarks and the matched Ames run.

## Session progress log

Append dated entries here after each work session. Record decisions, evidence,
artifacts, failures, blockers, and the next action; avoid narrative status without
a verifiable output.

### 2026-09-05 — submission planning and repository audit

- Reiterated the goal as a neural-GCM-inspired differentiable simulator for Mars,
  with the paper centered on present-day model credibility and learning utility.
- Audited the repository documentation, recent history, test inventory, staged
  datasets, benchmark tooling, and checked-in outputs.
- Confirmed that most intended deterministic operators exist, but their coupled
  seasonal validation and several column acceptance tests remain open.
- Confirmed that existing MCD results are explicitly transient diagnostics; copied
  representative metrics above without promoting them to validation results.
- Identified the missing learned component/data contract and matched Ames run as
  paper-critical gaps.
- Established September 7 (data), September 10 (configuration), September 17
  (results/model), September 20 (tables), and September 24 (PDF) as hard gates.
- Next action: complete official-deadline/ArcoMars/data-access verification, assign
  owners and compute, then run the current test suite and measured 10-sol benchmark.

### 2026-09-05 — executable test baseline

- Verified commit `8314609cac7fcd91c6a38dc8440392493c7eccc2` with Python
  3.12.3 and pytest 9.0.3.
- Focused command: `uv run --project package python -m pytest
  package/tests/gcm3d -m "not slow"`; result: **125 passed, 3 skipped,
  2 deselected in 74.43 s**.
- Full package command: `uv run --project package python -m pytest package/tests
  -m "not slow"`; result: **290 passed, 3 skipped, 5 deselected in 102.39 s**.
- The first wrapped test attempt misparsed the quoted marker and reported
  `file or directory not found: slow`; rerunning through the pass-through wrapper
  preserved the pytest expression. This was a command-wrapper issue, not a model
  failure.
- Interpretation: the implementation baseline is green, including the focused
  GCM suite, but this does not exercise the deselected long integrations or prove
  the open scientific validation requirements.
- Exclusion audit: the five slow tests are Held–Suarez climate, dry conservation
  across timestep/resolution, a 50-year intervention run, accurate-vs-fast cap
  balance, and the observed-band seasonal pressure swing. The three skips are
  stale JAX/Torch parity cases in `test_terraforming_ode.py` caused by the Torch
  model's move from one aggregate ice reservoir to separate north/south caps.
- Running the two slow GCM tests alone initially exposed a precision-ordering bug:
  Held–Suarez mass drift was `3.50e-5` versus the `2e-5` threshold, and resting
  drift was around `9.48e-7` versus the float64 `1e-12` threshold. The module had
  relied on another test importing first and globally enabling JAX float64.
- Added an explicit `jax_enable_x64` configuration to `test_benchmarks.py`. The
  isolated two-test run then passed in 53.25 s, and the complete five-test
  benchmark module passed in 33.48 s with a warm compilation cache.
- Next action: decide whether to re-port or remove the stale 0-D parity path and
  measure a canonical end-to-end run before committing scarce compute to
  Mars-year experiments.

### 2026-09-05 — canonical T21/12 stability and throughput screen

- Preserved all previous outputs and wrote only to new `outputs/iclr_smoke_*`
  directories. The established T21/8, 300 s result remains unchanged.
- Screened the complete `convection` ablation configuration with TES surface
  properties, prescribed dust, CO2 radiation and surface cycle, regolith,
  stability exchange, PBL diffusion, and dry convective adjustment.
- **Rejected:** T21/12 at 600 s became non-finite inside its first sol. The driver
  exited with `FloatingPointError` after 18.13 s and exported no valid result.
- **Accepted for continued screening:** T21/12 at 300 s completed one sol/296
  steps in 30.44 s including compilation and export, then resumed from its saved
  restart to ten sols/2,959 total steps. The nine-sol extension took 236.04 s,
  or **26.23 s per simulated sol** on the local CPU.
- Linear planning estimate at the measured steady rate: **4.87 hours per 668-sol
  Mars year** and **9.73 hours for one spin-up plus one evaluation year**, before
  contingency and downstream benchmarking. Reserve at least 15 hours per T21/12
  canonical configuration; multiple calibrations cannot each use full-year
  backpropagation on this machine.
- The ten-sol manifest, restart, NetCDF, and five diagnostic maps are present at
  `outputs/iclr_smoke_t21_l12_dt300/`. The manifest correctly labels the result
  as a transient rather than climatology.
- Decision: use short differentiable training windows and a longer forward-only
  validation rollout. Do not launch the two-year canonical run until the data
  target, fixed physics configuration, checkpoint cadence, and output sampling
  are frozen, because every rerun costs roughly half a day locally.
- Next action: inspect ten-sol physical/budget diagnostics, measure T42 throughput,
  and add elapsed-time/throughput metadata to future manifests so estimates no
  longer depend on shell timing logs.

### 2026-09-05 — ten-sol artifact and provenance audit

- Inspected the canonical smoke NetCDF rather than inferring success from process
  exit alone. All exported fields are finite. Ranges are: surface pressure
  **263.91–1104.62 Pa**, surface temperature **148.84–265.85 K**, horizontal wind
  speed **0.16–63.59 m/s**, and CO2 frost **0–55.38 Pa-equivalent**. These are
  plausible transient bounds, not agreement metrics.
- The artifact contains final 2-D pressure, surface temperature, near-surface
  winds, frost, and topography plus configuration metadata. It does **not** contain
  time-resolved TOA/surface energy, atmospheric-plus-cap CO2 mass, dry mass, AAM,
  soil drift, or equilibration diagnostics. Therefore the existing driver is not
  yet sufficient for the canonical two-year paper run even though its state is
  finite.
- Confirmed from the forcing path that `co2_radiation_enabled=True` and
  `ames_correlated_k_enabled=True`; the previous generic “two-stream CO2-band”
  label under-described the active path. Updated map provenance to distinguish
  **Ames correlated-k CO2 radiation**, compact-band two-stream radiation, and the
  grey fallback.
- Added invocation start step, executed steps, elapsed wall time, and simulated
  sols per wall hour to future ablation manifests. A three-step driver smoke test
  produced all four new fields and the corrected radiation label.
- Regression check after these changes: all **13** `test_maps.py` tests pass in
  6.98 s. `git diff --check` remains required at handoff.
- Decision: do not interpret the ten-sol ranges as a physics benchmark. Add a
  sampled budget/equilibration time series before launching the production
  spin-up, then use the final maps only as one view of the climate state.

### 2026-09-05 — T42 resolution-cost screen

- Ran the full T42/12 configuration at the driver-verified 450 s timestep in a
  separate output directory. A 0.1-sol/20-step cold-start screen completed in
  16.39 s inside the driver; all final fields were finite.
- Resumed the same state to one sol/197 total steps. The 177-step extension took
  117.86 s inside the driver and achieved **27.40 simulated sols per wall hour**.
- Linear estimate: **24.38 hours per T42 Mars year** and **48.75 hours for a
  spin-up/evaluation pair**, before contingency. Reserve at least 72 hours for
  one T42 pair on this CPU. This is about five times the measured T21/12 cost.
- Decision: T42 cannot be the training or ablation grid on the current local
  compute schedule. Use it for one forward-only resolution-sensitivity evaluation
  after the T21 model/data protocol is frozen, or move it to reserved accelerator/
  higher-throughput compute and remeasure there.
- The T42 result is only a one-sol transient and is evidence for execution cost
  and short stability, not climate accuracy or one-year stability.

## Source-of-truth links inside this repository

- Physics ledger: `docs/ideas/gcm3d-physics-limitations.md`
- Ames mapping: `docs/ideas/amesgcm-comparison.md`
- Architecture: `docs/package/gcm3d/architecture.md`
- Implementation: `docs/package/gcm3d/implementation.md`
- Data staging: `docs/scripts/gcm3d-data.md`
- MCD benchmark: `scripts/benchmark_mcd.py`
- Restartable Mars-year driver: `scripts/run_gcm3d_ablation.py`
