# Progress report

**Recorded:** 2026-09-15 16:35:38 PDT  
**Status:** ARCO-Mars data-source assessment and recommended neural experiment

## Current decision

Use ARCO-MACDA as the primary training source for the hybrid Mars GCM. Stream
only the required variables and Mars years from its Zarr v3 archive, begin with
daily means to match the model's current daily-mean solar forcing, and restrict
the first learned residual to bounded temperature and horizontal-wind
tendencies. Keep surface pressure, atmospheric mass, and CO2 mass outside the
learned correction until the physical model's CO2 conservation issue is fixed.

## What ARCO-Mars contains

ARCO-Mars combines three Mars reanalyses:

| Dataset | Mars years | Grid | Main advantage |
|---|---:|---:|---|
| ARCO-EMARS | MY 24-33 | 36 x 60 x 28 | Ensemble mean and uncertainty |
| ARCO-MACDA | MY 24-35 | 36 x 72 x 35 | Most variables, including 3-D dust and radiation |
| ARCO-OpenMARS | MY 24-35 | 36 x 72 x 35 | Useful second reanalysis comparison |

The datasets assimilate spacecraft measurements from TES, THEMIS, MCS, and
related instruments. They are reanalyses: observations combined with Mars GCM
simulations.

They are stored as Zarr v3 and can be read directly from Hugging Face without
downloading the complete archives.

## Best choice: ARCO-MACDA

ARCO-MACDA is the best training source for our model. It contains:

- 3-D atmospheric temperature
- 3-D east-west wind
- 3-D north-south wind
- Vertical pressure velocity
- Surface pressure
- Surface temperature
- Column dust opacity
- 3-D dust mixing ratio
- Surface CO2 ice
- Surface shortwave radiation
- Surface longwave radiation
- Geopotential
- Solar longitude and Mars-year information

Its complete shape is:

- 96,480 time steps
- 35 vertical levels
- 36 latitudes
- 72 longitudes
- Mars Years 24-35

The complete ARCO-MACDA archive is about 160 GB, but Zarr allows us to read only
selected years, variables, and time ranges. We should not download the whole
archive.

The number of states strongly indicates approximately 12 samples per sol,
although we should verify the exact time-coordinate spacing before building
training pairs.

- Dataset: [ARCO-MACDA](https://huggingface.co/datasets/ananyo01/ARCO-MACDA)
- DOI: [10.57967/hf/8771](https://doi.org/10.57967/hf/8771)

## Licensing

The Hugging Face ARCO-MACDA dataset is marked CC BY 4.0. Its underlying MACDA
data also records the UK Open Government Licence 3.0 and requires attribution to
the original dataset.

The arXiv manuscript itself is CC BY-SA 4.0. That manuscript license is separate
from the dataset license.

This is usable for an academic paper as long as we cite both ARCO-MACDA and the
original MACDA v2.0 dataset.

## Why it fits our model

The horizontal resolution is already close:

- ARCO-MACDA: 36 x 72
- Dinosaur T21: approximately 32 x 64

The main conversion is vertical:

- ARCO-MACDA: 35 sigma levels
- Our model: 12 levels

We can interpolate MACDA onto the Dinosaur grid and retain enough vertical
information for a small neural model.

ARCO-MACDA also includes consecutive atmospheric states. That makes it much
better than the Ames five-sol averages for training a neural tendency
correction.

## Recommended experiment

Use a small neural residual inside the GCM:

```text
Physical Dinosaur forecast
        +
Small neural correction
        =
Corrected forecast
```

The network should predict only:

- Temperature tendency
- East-west wind tendency
- North-south wind tendency

It should initially produce:

- No surface-pressure tendency
- No atmospheric-mass tendency
- No CO2 mass change
- Bounded temperature and wind corrections

This prevents the neural network from hiding the current CO2 conservation
problem.

## Training setup

A clean split would be:

- Training: MY 24-31
- Validation: MY 32-33
- Testing: MY 34-35
- Special out-of-distribution test: withhold MY 28, which contained a major
  global dust storm

We should also evaluate independently against Ames. MACDA and MCD are related to
the LMD Mars modelling family, so Ames provides a more independent model
comparison.

The simplest training pipeline is:

1. Read selected ARCO-MACDA variables from the cloud.
2. Convert them to daily means initially.
3. Interpolate from 35 levels to our 12 levels.
4. Regrid to the Dinosaur T21 grid.
5. Initialize the physical GCM from a MACDA state.
6. Run the GCM forward.
7. Measure its temperature and wind error.
8. Train the network to correct that error.
9. Test on Mars years and dust storms excluded from training.
10. Run longer rollouts to measure stability.

Using daily means matches our current daily-mean solar forcing. Later, we can
use the full subdaily data after validating the model's diurnal forcing.

## Revised effort

Because ARCO-Mars is already cloud-ready and well organized, the neural
experiment is easier than initially estimated:

| Work | Effort |
|---|---:|
| Verify time coordinate and stream a sample | 2-4 hours |
| Build ARCO preprocessing | 0.5-1 day |
| Vertical and horizontal regridding | 0.5 day |
| Add bounded neural residual | 0.5-1 day |
| Build one-step training | 1 day |
| Short-rollout training and stability | 1-2 days |
| Baselines, three seeds, figures | 1-2 days |

A credible first result should take approximately 4-6 intensive days, assuming
the physical model is stable.

## Final assessment

ARCO-Mars solves the largest data problem for the neural component. It provides
enough years, time steps, vertical levels, and physical variables to train and
test a small hybrid Mars GCM.

The strongest paper design is now:

1. Dinosaur/JAX differentiable Mars GCM.
2. Fix CO2 mass conservation.
3. Train a small temperature-and-wind residual using ARCO-MACDA.
4. Test on withheld Mars years and the MY 28 dust storm.
5. Compare with:
   - Physical GCM
   - Gradient-calibrated GCM
   - Neural residual GCM
   - Ames MGCM
   - MCD
6. Report mass conservation and long-rollout stability.

This gives the project a real neural component, a data-assimilated training
source, held-out evaluation, and two separate model references.

## Immediate next actions

1. Verify the exact ARCO-MACDA time-coordinate spacing by streaming a small
   sample.
2. Record the native coordinates, variable names, units, missing-value handling,
   and citations in a versioned dataset contract.
3. Fix and test CO2 mass conservation before enabling learned rollouts.
4. Implement daily aggregation, 35-to-12-level interpolation, and T21 horizontal
   regridding.
5. Add a bounded residual interface for temperature and horizontal winds.
6. Build one-step baselines before short-rollout training.
7. Preserve MY 34-35 and MY 28 from model selection and training.

## 2026-09-16 — branch baseline and ARCO-MACDA preparation

### CO2 conservation fix

- Identified the production failure: the phase-change tendency conserves
  `p_s + p_ice` instantaneously, but the dynamical integrator advances
  `log(p_s)`, so finite Runge-Kutta steps accumulated a false global mass
  source. The previous positivity projection did not remove that error.
- Added a Gaussian-quadrature global inventory projection after each complete
  timestep. It rescales atmospheric pressure while preserving its spatial
  pattern, retains non-negative frost, and subtracts configured atmospheric
  escape exactly.
- The production 10-sol gate changed combined atmospheric plus surface CO2 from
  `2.5984358235794388e16 kg` to `2.5984358234819876e16 kg`: a drift of
  `-974,512 kg`, or `-3.7503793288e-11` relative
  (`-3.7503793288e-9 percent`).
- The CO2 regression suite passes all 10 selected tests, including a test that
  verifies an explicit `10,000 kg s-1` escape sink is retained exactly.

### ARCO-MACDA verification and staging

- Verified the live Zarr v3 store rather than relying on its summary card.
- Confirmed 96,480 states, 35 sigma levels, 36 latitudes, 72 longitudes, and an
  exact cadence of 12 states per sol.
- The logical xarray size is `216,067,346,492 bytes` (about 216 GB), higher than
  the earlier 160 GB estimate.
- Found a partial 202-state MY36 tail even though complete documented years are
  MY24-35. Also confirmed that explicit Mars-year state counts vary, so the
  pipeline binary-searches `MY_Ls` instead of assuming a fixed year length.
- Added `scripts/stage_arco_macda.py` and staged finite 10-sol MY24 samples on
  the Dinosaur T21/L12 grid. The atmospheric artifact SHA-256 is
  `0a973c94bbb49a60b00e1fef305f19053e60a42db5fbf3b5a53f726414873c1d`;
  the surface-radiation artifact SHA-256 is
  `4d3620b514cd4c6e17487c73750f864a4a55cbb500c9d8d57ef3abb2e8e379b9`.
- Recorded schema, licenses, citations, year boundaries, processing, and the
  leakage-safe split in `docs/ideas/arco-macda-dataset-contract.md`.

### Physical baseline

- Configuration: T21, 12 sigma levels, 300 s timestep, daily-mean solar
  forcing, MOLA orography, TES albedo and thermal inertia, Ames seasonal dust,
  Ames correlated-k CO2 radiation, multilayer regolith, Richardson exchange,
  PBL diffusion, dry convective adjustment, and the corrected energy-limited
  CO2 cycle.
- The 10-sol production gate completed with finite exported fields. Surface
  temperature was `145.62-237.53 K`; surface pressure was
  `251.42-1246.24 Pa`; wind speed was `0.063-34.73 m s-1`.
- A fresh 668-sol baseline is checkpointing under
  `outputs/branch_physical_baseline_t21_l12_dt300/`. Treat it as a numerical
  stability and first-year spin-up result, not proof of climatological
  equilibrium.
- Added `scripts/evaluate_physical_baseline.py` to produce MOLA maps,
  pressure-topography statistics, ARCO-MACDA difference maps, Gaussian-weighted
  error metrics, exact mass drift, artifact hashes, and explicit limitations.

## 2026-09-16 21:19:55 PDT — long-run stability diagnosis

- The original unfiltered 300 s baseline did **not** complete. It remained
  finite through step 151,559 (`512.1664323438 sols`, `Ls=268.423 deg`) and
  became non-finite during the next 30-step block. The earlier coarse guard had
  first located the failure before step 152,439; replaying from the valid
  checkpoint at 0.1-sol resolution narrowed the onset to before step 151,589.
- Combined atmosphere-plus-frost CO2 at the last valid unfiltered checkpoint
  was `2.598435822393495e16 kg`. Relative to the initial
  `2.5984358235794388e16 kg`, the drift was `-4.5633063103e-10`
  (`-4.5633063103e-8 percent`). This confirms the CO2 inventory projection
  remained effective through the failure.
- The failure was a dynamical blow-up. At the last valid checkpoint, decoded
  air temperature spanned `118.861-379.859 K`, and maximum horizontal wind
  speed was `2953.397 m s-1`. These values are physically unusable even though
  all state arrays were still finite.
- Root cause: the spectral primitive-equation integration had no horizontal
  scale-selective dissipation, allowing kinetic energy/enstrophy to accumulate
  at the truncation scale. Added standard fourth-order spectral horizontal
  diffusion with a highest-wavenumber e-folding time of `0.1 sol`. The filter
  applies only to modal atmospheric fields; it leaves nodal surface
  temperature, frost, and soil reservoirs unchanged. The CO2 inventory
  projection remains the outermost post-step operation.
- A recovery experiment starting from the already degraded 512.166-sol state
  crossed the exact prior failure point and completed at 515.498 sols. In
  3.332 sols, maximum wind speed fell from `2953.397` to `46.010 m s-1`, air and
  surface fields remained finite, final surface pressure was
  `248.097-1045.534 Pa`, and final surface temperature was
  `149.043-271.746 K`.
- The recovery experiment demonstrates that the missing dissipation caused the
  immediate crash, but it is not accepted as the production baseline because
  its first 512 sols contain an unphysical high-wind episode. A fresh 668-sol
  run with diffusion active from initialization is required before claiming a
  stable physical baseline or reporting final MOLA/ARCO error statistics.

## 2026-09-16 21:45:39 PDT — clean filtered production run active

- Started a fresh production run from the resting MOLA-balanced initial state
  in `outputs/branch_physical_baseline_t21_l12_dt300_hyperdiff01/`. Its physical
  configuration matches the failed run, with fourth-order horizontal diffusion
  (highest resolved wavenumber e-folding time `0.1 sol`) added from step zero.
- Extended checkpoint diagnostics to record minimum/maximum atmospheric
  temperature and maximum three-dimensional wind speed. These fields exposed
  the unfiltered blow-up before NaNs and are now part of the stability evidence.
- At the latest recorded checkpoint, step 11,840 (`40.0111544610 sols`), all
  fields were finite. Surface temperature was `144.590-235.337 K`, atmospheric
  temperature was `108.411-227.846 K`, and maximum wind speed was
  `115.799 m s-1`.
- Combined atmosphere-plus-frost CO2 at that checkpoint was
  `2.5984358232419816e16 kg`, a relative drift of `-1.2986935804e-10` from the
  exact initial inventory. This remains many orders of magnitude smaller than
  the original mass-conservation failure.
- This run is still in progress. No final stability, MOLA comparison, or ARCO
  error claim is valid until it reaches all 668 requested sols and the exported
  artifacts pass the final evaluator.

## 2026-09-17 10:16:36 PDT — completed baseline and final evaluation

### Completion and numerical stability

- The clean filtered baseline completed all 197,673 requested steps:
  `668.0004168718 sols`, ending at `Ls=359.682671 deg`. Every exported field and
  every value in all 135 checkpoints is finite.
- Final surface ranges were pressure `221.822-1078.764 Pa`, surface temperature
  `144.327-241.221 K`, near-surface wind speed `0.203-30.017 m s-1`, and CO2
  frost `0-2845.423 Pa-equivalent`. The final checkpoint's maximum wind over
  all 12 atmospheric levels was `129.729 m s-1`.
- Combined atmospheric plus surface CO2 changed from
  `2.5984358235794388e16 kg` to `2.598435820403596e16 kg`: exactly
  `-31,758,428 kg`, or `-1.2222132912e-9` relative
  (`-1.2222132912e-7 percent`). This is the maximum absolute checkpoint drift.

### MOLA topology result

- MOLA elevation on the T21 grid spans `-7260.527` to `14478.380 m`, with an
  area-weighted mean of `-560.377 m` and standard deviation `2892.316 m`.
- Final surface pressure has correlation `-0.960459` with elevation, a fitted
  linear slope of `-52.582 Pa km-1`, and a fitted log-pressure scale height of
  `10,951.313 m`. The map clearly resolves high pressure in Hellas and low
  pressure over Tharsis/Olympus, within T21 smoothing.

### ARCO-MACDA differences

The final instantaneous model state at `Ls=359.683 deg` was compared with the
MY24 ARCO daily mean at `Ls=0.278 deg`; the circular seasonal mismatch is
`0.595 deg`. Metrics use Gaussian spherical-area weights.

| Field | Bias | MAE | RMSE | Relative MAE | Spatial correlation |
|---|---:|---:|---:|---:|---:|
| Surface pressure | `-14.121 Pa` | `23.575 Pa` | `29.714 Pa` | `3.905%` | `0.987174` |
| Surface temperature | `+13.397 K` | `13.403 K` | `15.051 K` | `6.596%` | `0.953123` |
| Eastward wind | `-1.163 m s-1` | `5.482 m s-1` | `7.583 m s-1` | `112.808%` | `0.469017` |
| Northward wind | `-1.382 m s-1` | `5.456 m s-1` | `7.161 m s-1` | `177.752%` | `0.361018` |
| Wind speed | `+3.447 m s-1` | `4.706 m s-1` | `5.964 m s-1` | `74.112%` | `0.454818` |
| CO2 frost | `-47.328 Pa-eq` | `54.740 Pa-eq` | `231.449 Pa-eq` | `42.937%` | `0.888336` |

Pressure and the broad temperature structure are strong first-baseline results.
The model is too warm by about `13.4 K`, winds have weak spatial agreement and
too much mean speed, and the seasonal CO2 frost reservoir is too small. These
are physical-model errors, not numerical instability.

### Artifacts and scope

- Machine-readable report:
  `outputs/branch_physical_baseline_t21_l12_dt300_hyperdiff01/evaluation/baseline_report.json`
- MOLA/pressure figure:
  `outputs/branch_physical_baseline_t21_l12_dt300_hyperdiff01/evaluation/mola_topography_pressure.png`
- ARCO comparison and difference figure:
  `outputs/branch_physical_baseline_t21_l12_dt300_hyperdiff01/evaluation/arco_macda_comparison.png`
- Model NetCDF SHA-256:
  `614103d9cbfe94f48f8b4bcab7c68dd649e51c9b95be6fa7dd0de9a5fec3d3bd`
- Diagnostics SHA-256:
  `0244018c726574e63d76e66449ead50c2f9e6f00cb74a155a656cb2fd7214e00`
- Verification: all 42 relevant GCM physics/submission tests pass and
  `git diff --check` reports no whitespace errors.

This first Mars year establishes numerical stability, not climatological
equilibrium. ARCO-MACDA is a reanalysis rather than direct truth, its sample is
a daily mean while the model output is instantaneous under daily-mean forcing,
and this MY24 sample is a pipeline comparison rather than a held-out test score.

## 2026-09-17 15:41:28 PDT — energy-bias correction and equilibrium run

### Diagnosed causes

- The original `+13.397 K` ARCO and `+10.020 K` Ames year-end temperature
  biases combined a real radiative error with an unfair temporal comparison.
  Daily-mean sunlight suppresses the strong Martian day/night cycle; because
  emission scales as `T^4`, this raises the model's mean surface temperature.
- The Ames correlated-k solar path also omitted the documented zenith-angle
  air-mass factor. Added instantaneous path scaling for diurnal forcing and the
  analytic flux-weighted daily path cosine `<mu^2>/<mu>` for daily forcing.
- The correlated-k thermal path ignored the prescribed Ames longwave dust field
  and inferred IR dust entirely from visible opacity. It now uses the separate
  longwave opacity normalized to the Ames IR reference band.
- At `Ls=4.775 deg`, the corrected model's surface shortwave-down flux was
  `132.479 W m-2`; ARCO MY24 gave `137.072 W m-2`. Excess surface sunlight was
  therefore not the remaining warm-bias source. Model downwelling longwave was
  `22.583 W m-2`, versus `11.160 W m-2` in low-dust ARCO (`tau_vis=0.031`).
  Separate CO2 and dust longwave scale controls were added so this calibration
  is explicit and reproducible.

### Calibration probes

- A 10-sol diurnal/full-opacity probe from the exact year-one restart reduced
  the matched Ames temperature bias to `+6.931 K`, with `7.045 K` MAE and
  `0.971589` spatial correlation after averaging five local-time phases.
- A 10-sol diurnal probe with both longwave opacity scales set to `0.25` reduced
  the matched Ames bias to `+4.279 K`, MAE to `4.705 K`, and retained
  `0.972040` spatial correlation. Its fixed-state TOA imbalance was
  `+0.433 W m-2`, small enough to proceed to a full seasonal test.
- These are calibration probes, not equilibrium claims. The Ames dust scenario
  has mean visible opacity near `0.35` at this season, while MY24 ARCO has only
  `0.031`; dust opacity was therefore not tuned directly to the ARCO longwave
  flux.

### Corrected production experiment

- Started two complete corrected Mars years from the already spun-up 668-sol
  restart under `outputs/branch_physical_diurnal_lw025_t21_l12_dt300/`.
  The exact orbital period is `668.618832520472 sols`, so the run target is
  total sol `2005.238081912778`: the first corrected year is adjustment, and
  the two corresponding year-end states provide the seasonal convergence test.
- Release configuration: T21/L12, `dt=300 s`, resolved diurnal forcing,
  zenith-correct solar paths, separately prescribed visible/IR dust,
  `0.25` CO2 and dust longwave opacity scales, and fourth-order spectral
  diffusion with a `0.1-sol` highest-mode e-folding time.
- At the first production checkpoint (`673.001811 sols`), all fields were
  finite, mean surface temperature was `209.577 K`, maximum three-dimensional
  wind was `129.812 m s-1`, and combined CO2 was
  `2.598435820375651e16 kg`.
- Verification currently passes all 45 focused physics/submission tests,
  compilation checks, and `git diff --check`. Merge remains gated on completion
  of the multi-year convergence and seasonal benchmark.

## 2026-09-18 — corrected convection run completed

- The diurnal, longwave-corrected convection run reached its exact target at
  step `593385`, sol `2005.2380819402`; the runner exited and wrote the final
  restart, NetCDF, maps, plots, and checkpoint.
- Final mean surface pressure was `576.597 Pa`, mean surface CO2 frost was
  `93.075 Pa-equivalent`, and mean surface temperature was `209.839 K` with a
  `144.315-268.195 K` surface range. Maximum three-dimensional wind was
  `118.679 m s-1`; all recorded fields were finite.
- Combined atmosphere-plus-frost CO2 changed by `-46,959,948 kg` over the two
  corrected years. Maximum absolute relative drift was `1.8072391e-9`.
- The declared global seasonal convergence evaluator passed pressure, surface
  CO2 frost, and surface temperature at Ls `45`, `135`, `225`, and `315`
  degrees. Maximum year-over-year changes were `2.497 Pa`, `2.497 Pa`, and
  `0.620 K`, within thresholds of `5 Pa`, `5 Pa`, and `1 K`.
- The overall gate failed only on mean deep-soil temperature. Its seasonal
  year-over-year changes were `0.442-0.634 K`, above the `0.25 K` threshold.
  The atmosphere/surface seasonal cycle is repeatable by these global metrics,
  but the regolith is not yet equilibrated.
- Machine-readable report:
  `outputs/branch_physical_diurnal_lw025_t21_l12_dt300/evaluation/seasonal_convergence.json`.

## 2026-09-19 — deterministic paper pipeline and `tform serve`

- Removed the neural-residual experiment from the paper scope. The implemented
  contribution is a deterministic differentiable Mars GCM with three bounded,
  interpretable calibration controls: CO2 longwave opacity, dust longwave
  opacity, and bulk surface exchange.
- `tform serve` now launches the corrected physical configuration by default:
  CO2 and dust longwave scales `0.25`, surface-exchange multiplier `1.0`,
  correlated-k CO2 radiation, seasonal Ames visible/IR dust, TES albedo and
  thermal inertia, regolith, stability-dependent exchange, PBL diffusion, dry
  convection, MOLA orography, and the conservative CO2 cycle. The three
  controls are available in an optional UI panel and validated against the same
  fixed calibration bounds used by the command-line optimizer.
- Existing UI outputs remain intact: surface temperature, pressure, zonal and
  meridional wind, wind speed, CO2 frost, and MOLA elevation. Physical
  parameters and forcing provenance are saved in map metadata and exported
  NetCDF attributes.
- Added a full-state comparison schema, gap-aware ARCO staging, a coupled JAX
  calibration driver with an equal 20-call L-BFGS/Powell protocol, and seasonal
  ARCO, Ames, and MCD evaluators. Reverse-mode timestep rematerialization bounds
  calibration memory for the two-sol rollout.
- Held-out ARCO-MACDA MY34--35 diagnostics cover four seasons and all 12 model
  levels. Air-temperature MAE is `6.22-10.28 K` with correlation
  `0.871-0.964`; eastward-wind correlation is `0.871-0.939`; northward wind and
  seasonal frost remain deficient.
- NASA Ames four-season diagnostics give surface-temperature bias
  `+2.36` to `+4.74 K` and pressure correlation `0.985-0.987`. Near-surface
  winds remain too fast, with speed bias `7.53-12.39 m/s`.
- Four MCD v6.1 seasonal comparisons were cached at local times 0, 6, 12, and
  18 h. Temperature MAE is `15.07-19.53 K`, pressure correlation is
  `0.971-0.980`, and wind-speed correlation is `0.215-0.458`. These are
  quantitative model-reference diagnostics, not reproduction or observational
  validation claims.
- Fixed boolean NetCDF attribute serialization in both server exports and the
  MCD native cache. The UI production build passes; 12 focused server tests and
  61 physics/maps/submission tests pass. The revised 13-page manuscript builds
  successfully.
- One additional orbital period is running from the corrected restart to test
  the declared `0.25 K` deep-soil convergence threshold. The full two-sol,
  20-evaluation calibration is also running; neither result is claimed before
  its machine-readable report completes.

## 2026-09-19 — short-horizon GPU workload prepared

- Supersedes the two-sol calibration configuration above: the physical tangent
  rollout is too expensive on the laptop. The frozen experiment now requests
  0.25 sol, rounded to 74 physical steps at 300 s, with T21/L12 and all operators
  retained. This is **short-horizon coupled calibration**, not equilibrium tuning.
- Added `apps/mars-calibration`, an installable application depending on
  `terraforming[gcm3d]`, with a CUDA 12 JAX container, Nautilus Kubernetes Job,
  persistent-volume transfer workflow, and a fixed `experiment.json`.
  The original script remains a compatibility entry point.
- Uses the existing MY32 Ls45 target index 5 and checkpoint 477393, frozen with
  TES, Ames dust, and MOLA in `outputs/nautilus-calibration-inputs/`. Every input
  has a SHA-256 manifest; no data downloads happen in the GPU job.
- Enforces exactly 20 L-BFGS-B and 20 Powell oracle calls; an early optimizer
  termination restarts from the best evaluated point, with restart counts
  disclosed. Twelve centered finite-difference forward calls are separate.
  The initial autodiff gradient is reused from optimization. Forward mode now
  obtains the primal loss as auxiliary output of the same differentiated call.
- CUDA fallback to CPU is rejected. Per-call JSONL logs, package versions,
  device identity, input hashes, and final acceptance report persist on the PVC.
- The target is a daily mean and the model endpoint is instantaneous. This
  single-window demonstration does not establish MY33 validation, four-season
  calibrated performance, or equilibrium sensitivity. Existing MY34–35
  diagnostics remain physical-baseline diagnostics.
- Local application tests pass for exact budget enforcement, early termination,
  finite-difference accounting, input-integrity checks, and canonical settings.
  The input bundle passes preflight. Container build and the 74-step CUDA
  experiment remain to be run on Nautilus; no GPU calibration result is claimed.
- A real one-step coupled CPU smoke run completed with exact 1/1 oracle
  counts and all six finite-difference checks passing (maximum relative error
  `6.1471e-6`). Its overall gate correctly failed because a single initial
  evaluation cannot demonstrate loss reduction. This is integration verification,
  not the requested 74-step calibration result. Report:
  `outputs/nautilus-calibration-smoke/report.json`.

## 2026-09-19 — common reference grids and AmesCAP export

- Added `tform mars compare` with an explicit cached-input JSON configuration.
  Generated an offline HTML report for Ls45/135/225/315 with physical-model,
  Ames, MCD and ARCO-MACDA surface maps, common row color ranges, centered
  difference maps, MOLA terrain, parameter/sampling tables, and CSV/JSON metrics.
- Uses Gaussian area weights on T21, periodic longitude interpolation, explicit
  time/level selection, and unit-checked frost conversion. Missing fields are
  unavailable; unresolved dimensions and non-finite fields fail explicitly.
- The default compares the physical baseline only. Sparse model snapshots,
  Ames five-sol means, ARCO daily means, and four-time MCD averages are not
  perfectly matched. Wind-height/dust differences and monthly MCD frost are
  limitations; no equilibrium or calibrated-climate improvement is claimed.
- Added AmesCAP 2-D NetCDF exports and a multi-source MarsPlot template.
  Actual MarsPlot 3.3 rendering succeeded for all 28 Ls45 panels (eight pages).
  Current MarsPlot 3.5 additionally requires the user's missing
  `~/.amescap_profile`; its complete rendering is not claimed.
- Application comparison tests and existing CLI command tests pass (8 tests).
  Artifacts: `outputs/reference-comparison/index.html`, `metrics.csv`,
  `parameters.csv`, `mola.png`, and `amescap-rendered/plots/`.
