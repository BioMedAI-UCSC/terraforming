# Differentiable Mars model: experiment runbook

The claim is a differentiable physical Mars GCM with neural-component support.
The new neural experiment inserts the framework's bounded temperature-heating
MLP into the coupled physical timestep. It is not a PBL closure or an offline
forecast correction. It does not enforce zero net neural heating; explicitly
report this energy-source limitation. Zero-output initialization must reproduce
the physical forecast and a random-direction neural derivative must agree with
centered finite differences before fitting starts.

Run commands from the repository root, using the installed environment. GPU
commands assume GPU-enabled JAX. No jobs are submitted by this runbook. Use a
fresh output directory for every command; completed artifacts are never silently
overwritten. The new training runner writes checkpoints for inspection but does
not yet resume training; use short pilots before expensive campaigns.

## 1. Check installation and smoke-test the coupled paths

```bash
rtk proxy .venv/bin/python -m pytest apps/mars-calibration/tests package/tests/gcm3d/test_neural_temp.py -q
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py gradients --inputs outputs/nautilus-calibration-inputs --output outputs/differentiable/gradient-smoke --steps 1 --epsilons 1e-3 1e-4
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py prepare-synthetic --inputs outputs/nautilus-calibration-inputs --output outputs/differentiable/synthetic-smoke --steps 1
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py neural --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/synthetic-smoke/windows.json --output outputs/differentiable/neural-smoke --updates 1 --epsilons 1e-3 1e-4
```

Synthetic windows are separated segments from a common parent simulation.
They test software and inverse-learning capability, not independent climates or
real Mars forecast skill. Teacher physical parameters are (0.6, 0.8, 1.4), while
the fitted neural model keeps physical parameters at (1, 1, 1). The target is
therefore a same-model physical discrepancy, not an arbitrary prescribed neural
answer. Validation selects the best checkpoint, including the original physical
baseline. Test outcomes are reported even if validation does not improve.

## 2. Physical gradient agreement versus horizon

```bash
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py gradients --inputs outputs/nautilus-calibration-inputs --output outputs/differentiable/gradients --steps 1 4 16 32 74 148 --epsilons 1e-2 1e-3 1e-4 1e-5 --require-gpu
```

Outputs: `gradients.csv`, `acceptance.json`, source hashes and environment.
The scalar objective is final area/layer-mean air temperature. Derivatives are
with respect to all three physical controls, in physical parameter coordinates.
The runner reports every epsilon, absolute and relative discrepancies, AD/FD
values, synchronized warm derivative time and compilation time. A finite
gradient alone is insufficient. `complete.json` means execution completed;
inspect `all_checks_pass` separately. Long horizons may legitimately fail at
switches or lose useful sensitivity. This experiment does not certify neural
reverse-mode gradients at every horizon; neural fitting separately checks a
random parameter direction at its actual training horizon.

## 3. Prepare aligned real-reference windows

Use the existing native MACDA staging pipeline (CPU/network work):

```bash
rtk proxy .venv/bin/python scripts/stage_mola.py
rtk proxy env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python scripts/neural_temp_data.py --run-dir outputs/differentiable/native --mola data/mola/meg004/megt90n000cb.img
rtk proxy .venv/bin/python scripts/prepare_differentiable_windows.py --native-run outputs/differentiable/native --output outputs/differentiable/macda --per-split 4
```

The first command is unnecessary if MOLA is already staged. Native staging
uses its existing fixed 112/32/32 coverage contract and can download substantial
data; the adapter then selects four deterministic seasonally distributed pilot
windows per split without consulting forecast errors. Increase `--per-split`
before fitting for a larger experiment. Do not select windows after seeing test
performance. Native contract/source changes require a new staging directory.

The adapter reuses observed T/u/v/pressure initialization, past-only soil spinup,
MACDA clock alignment, start-time dust and native targets. It saves per-window
forcing so the orbit and dust remain consistent. The runner uses native interval
/ 25 as dt (about 295.91748 s), automatically from the manifest. It never rounds
the target timestamp to 300 s. The physical runner retains its TES boundary
maps and fixed physical closure intervals; the reused soil spinup uses the
initializer's scalar thermal inertia and remains an initialization approximation.
Both model classes share it. Terrain checksums must match the input bundle.

## 4. Real-reference physical calibration and neural capability

```bash
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py calibration --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/macda/windows.json --output outputs/differentiable/calibration --updates 50 --require-gpu
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py neural --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/macda/windows.json --output outputs/differentiable/neural-seed0 --updates 50 --seed 0 --require-gpu
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py neural --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/macda/windows.json --output outputs/differentiable/neural-seed1 --updates 50 --seed 1 --require-gpu
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py neural --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/macda/windows.json --output outputs/differentiable/neural-seed2 --updates 50 --seed 2 --require-gpu
```

Start with `--updates 1` and a fresh pilot output on your GPU to measure compile,
memory and execution costs. All training windows enter each update; reverse-mode
cost grows with windows and horizon. Physical fitting uses forward-mode AD in
bounded normalized coordinates; neural fitting uses reverse-mode AD, an 8-unit
hidden layer, and bounded +/-2e-4 K/s heating. Both use Adam and norm clipping at
one; this is not the older L-BFGS-B/Powell comparison. Physical controls remain
fixed for neural fitting. Training targets determine field scales and training
start states determine feature normalization. All models use identical windows.

Outputs: `gradient-checks.csv`, `training.csv`, `checkpoint.npz`, `skill.csv`,
`acceptance.json`, `provenance.json`, `complete.json` or `failure.json`.
Skill contains T/u/v RMSE by window, split and lead time for physical and fitted
models. Weighting matches the existing paper runner: area weights and uniform
sigma-layer weights (equal mass fractions on this grid). Reports retain block
IDs. Multiple seeds do not make the same test windows independent. The plotter
reports descriptive errors, not inferential confidence intervals. Calibration
against MACDA is reanalysis calibration, not independent observational validation.

## 5. Paired ablations across initial states

```bash
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py ablations --inputs outputs/nautilus-calibration-inputs --manifest outputs/differentiable/macda/windows.json --output outputs/differentiable/ablations --cases no_pbl no_convection no_regolith half_timestep --require-gpu
```

Each case resets the same restart and forcing as its paired full run. The full
case is always included. All supplied windows are evaluated; this is sensitivity
analysis, not training or test-driven model selection. Plots show every start
and mean +/- sample SD, not confidence intervals. For an initial paper experiment
use at least four seasonally separated starts; the default adapter supplies 12.
The runner fails loudly on nonfinite cases rather than silently discarding them.

## 6. Multi-year climate and resolution/layer runs

The existing checkpointed physical runner already supports duration, resolution,
layers, timestep and resume. This command explicitly selects all its physical
processes (`convection` is the cumulative full-physics configuration):

```bash
rtk proxy .venv/bin/python scripts/run_gcm3d_ablation.py data/tes/mgs_tes_surface_1deg.nc --output-dir outputs/differentiable/climate-run --config convection --truncation T21 --layers 12 --dt 300 --initial-ls 0 --diurnal --sols 2006 --chunk-sols 5 --cooldown-seconds 0 --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc
```

Run on your GPU host and inspect the manifest backend. Resume with the same
command plus `--resume`; extend `--sols` if deep-soil convergence fails. Do not
claim equilibrium because three years elapsed. The baseline's initialization and
physical configuration differ from the recovery bundle; record that distinction.
Analyze the resulting checkpoint diagnostics:

```bash
rtk proxy .venv/bin/python scripts/analyze_differentiable_experiments.py climate --data outputs/differentiable/climate-run/convection/checkpoint_diagnostics.csv --output outputs/differentiable/climate-analysis
```

This analyzer requires
coverage from initialization, at least two complete years, monotone finite
samples and no gap longer than 20 sols. It compares matched phase bins in the
last two complete years; pressure/frost thresholds are 5 Pa, surface T 1 K and
deep-soil T 0.25 K. These are diagnostic criteria, not observational error bars.

For scaling, repeat the physical command with fresh directories and explicit
configurations such as T21/L8/dt300, T21/L12/dt300 and T42/L12/dt150. First verify
each configuration for a short pilot (`--sols 1`). Keep forcing, physical time,
precision and evaluation samples fixed; select stable timesteps rather than
assuming T42 tolerates 300 s. Evaluate their fields with the same matchup
operator below. Record repeated synchronized *matched forecast* timings in
`timing.csv`; whole-campaign wall times are not comparable if durations differ.

## 7. Observational validation and the accuracy-cost frontier

Acquiring Viking/TES/MCS observations and implementing a retrieval-specific
observation operator are external-data tasks, not completed by this runner.
The analyzer accepts a strict matched-data table after that step. Required CSV
columns:

`configuration,field,units,sample_id,block,observed,predicted,weight`

Use one row per actual matched observation; configurations must have identical
sample IDs, observed values, weights and temporal blocks. Never use grid cells
as independent replicates. The metadata JSON requires nonempty `source`,
`reference_kind`, `observation_operator`, `split`, `quality_control`, `units`,
and `time_alignment`. For observational validation, `reference_kind` must be
`observation` and `split` must be `test`. Document pressure/elevation matching
for Viking and pressure/altitude/local-time sampling and retrieval smoothing
for TES/MCS. Exclude fitting observations and assimilated overlap when claiming
independence. MOLA and TES boundary maps are not held-out atmospheric truth.

```bash
rtk proxy .venv/bin/python scripts/analyze_differentiable_experiments.py observations --data data/validation/matched-observations.csv --metadata data/validation/matched-observations.json --output outputs/differentiable/observations
rtk proxy .venv/bin/python scripts/analyze_differentiable_experiments.py scaling --data data/validation/matched-resolutions.csv --metadata data/validation/matched-resolutions.json --timing data/validation/timing.csv --output outputs/differentiable/scaling
```

These paths are user-prepared inputs, not files shipped by this change.
Scaling timing CSV requires `configuration,median_seconds`; document hardware,
warmup, repeats, dispersion, precision, simulated duration, output cadence and
compilation separately in metadata. Accuracy can use reanalysis if labelled.

## 8. Consistent paper figures

```bash
rtk proxy env MPLCONFIGDIR=/private/tmp/terraforming-mpl XDG_CACHE_HOME=/private/tmp/terraforming-cache .venv/bin/python iclr-results/figures/make_differentiable_figures.py --runs outputs/differentiable/gradients outputs/differentiable/calibration outputs/differentiable/neural-seed0 outputs/differentiable/ablations --out iclr-results/figures/out
rtk proxy env MPLCONFIGDIR=/private/tmp/terraforming-mpl XDG_CACHE_HOME=/private/tmp/terraforming-cache .venv/bin/python iclr-results/figures/make_differentiable_figures.py --runs outputs/differentiable/climate-analysis outputs/differentiable/observations outputs/differentiable/scaling --out iclr-results/figures/out
```

Only include directories whose experiments you have run. The plotter imports
the existing `plotstyle.py`, writes vector PDF plus 300-dpi PNG, and annotates
source artifacts and claim limitations. It does not aggregate different seeds
as independent observational samples. The full T/u/v metrics remain in CSV;
the compact rollout and ablation figures currently display temperature.
