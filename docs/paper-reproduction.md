# Reproduce and revise the AEGIS paper

Paper of record: [arXiv 2610.04081v1](https://arxiv.org/html/2610.04081v1).
Work tracker: [#104](https://github.com/BioMedAI-UCSC/terraforming/issues/104).
This guide separates rebuilding the evidence already available from running new
experiments. A completed script is not evidence that its scientific acceptance
criteria passed. Never replace a historical number with a new run silently.

## 1. Start from the fixes and install once

Merge the tools PR #115 and GHG-unit PR #116, then the companion manuscript PR.
The main workspace stays on `main`; development changes live on PR branches.
From the repository root:

```sh
rtk git switch main
rtk git pull --ff-only
rtk proxy uv sync --frozen --all-packages --dev --extra gcm3d --extra arco
```

Use `.venv/bin/python` below. `arco` supplies the HTTPS/Zarr dependencies;
`gcm3d` alone does not. Commands work on CPU; long climate and training runs
should use a GPU. On a supported Linux NVIDIA host, install the matching pinned
CUDA build once and verify the backend:

```sh
rtk proxy uv pip install --python .venv/bin/python 'jax[cuda12]==0.11.0'
rtk proxy .venv/bin/python -c 'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend()=="gpu"'
```

Do not run `uv sync` again during a CUDA experiment: it can restore CPU wheels.
Use a new output directory for each source revision, seed, and configuration.
Do not load historical nondimensional restarts into the changed sidereal units.

## 2. Rebuild the corrected paper now (no external climate downloads)

```sh
rtk proxy .venv/bin/python paper/arxiv-2610.04081/regenerate.py
rtk proxy latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/arxiv-2610.04081/revision/iclr2027_conference.tex
```

Open `paper/arxiv-2610.04081/revision/iclr2027_conference.pdf`.
`regenerate.py` verifies the original source archive and frozen evidence hashes,
checks repeated fixed-update results/checkpoints and citation keys, and builds
the reference table, radiation table, and radiation figure. Historical figures
remain historical; the missing Ls225 panel comes from the tracked seasonal
archive. `source.json` identifies the immutable original arXiv source archive.
`claims.json` maps every result group to its evidence and remaining work.

To repeat the historical small-data audit independently:

```sh
rtk proxy .venv/bin/python scripts/audit_paper_results.py --data-root paper/arxiv-2610.04081/evidence/historical-audit/inputs --output outputs/paper/review-audit --skip-fields
```

This reproduces the old calibration/control-vector and reference-bias analysis,
not a new fit or a corrected-model seasonal evaluation. The original field audit
needs `outputs/Untitled/full.nc`, `outputs/Untitled/config.json`,
`outputs/paper-year-gpu3/half_timestep.nc`, and that run's `ablations.json` under
`--data-root`. Those large fields are **not publicly bundled** here. Without them,
use the frozen `evidence/historical-audit/results.json`; do not claim to have
rerun its wind/Mach/frost-point or 300/150 s field calculations.

## 3. Reproduce the new radiation result (self-contained)

```sh
rtk proxy .venv/bin/python examples/neural/train_radiation.py --output outputs/paper/radiation-seed0 --seed 0 --epochs 100 --fine-tune-epochs 20 --continuation-updates 20 --continuation-budget updates
rtk proxy .venv/bin/python examples/neural/train_radiation.py --output outputs/paper/radiation-repeat --seed 0 --epochs 100 --fine-tune-epochs 20 --continuation-updates 20 --continuation-budget updates
```

Read each `report.json`: both continuations must have 20 updates, zero failed
columns, and a completed short coupled check. At 48 steps the archived seed-0
interpolation/higher-dust RMSE values are local **0.619738/1.798794 K**, trajectory
**0.566059/1.653355 K**, and continued local **0.591726/1.595449 K**. Check repeated
values and checkpoint arrays within numerical tolerance; bitwise GPU/CPU identity
is not required. Histories, predictions and checkpoints are saved by the script.
Use seeds 1 and 2 in separate directories before making multi-seed claims.
`--continuation-budget wall-time` is a separate machine-dependent efficiency
experiment; it must not replace the primary fixed-update table.

The controlled scalar inverse example is also self-contained:

```sh
rtk proxy .venv/bin/python examples/neural/recover_parameter.py --output outputs/paper/column-recovery --iterations 120 --steps 24 --seed 0
```

This recovers a generated column-model multiplier of 1.6. It is **not** the
reanalysis fit or evidence that (0.6,0.8,1.4) is a fitted Mars parameter vector.
For the separate neural-control example, the corrected metric is the final
uncontrolled forecast, not the initial field:

```sh
rtk proxy .venv/bin/python examples/neural/optimize_atmospheric_control.py --output outputs/paper/control --iterations 40 --steps 8 --layers 2 --target-shift-k -0.2 --seed 0
```

## 4. Prepare fresh physical/reanalysis inputs

Run the staging scripts from the checkout whose physics will be evaluated:

```sh
rtk proxy .venv/bin/python scripts/stage_mola.py
rtk proxy .venv/bin/python scripts/stage_tes_surface.py
rtk proxy .venv/bin/python scripts/stage_arco_macda.py --mars-year 32 --center-ls 45 --sols 10 --truncation T21 --layers 12 --output data/arco_macda/macda_my32_ls045_10sol_daily_t21_l12.nc
rtk proxy .venv/bin/python scripts/neural_temp_data.py --run-dir outputs/paper/native --mola data/mola/meg004/megt90n000cb.img
rtk proxy .venv/bin/python scripts/prepare_differentiable_windows.py --native-run outputs/paper/native --output outputs/paper/windows --per-split 4
```

Native staging pins ARCO-MACDA revision
`65a0bebd804b9c240752277e83f5737d58c6ee9c`, records source units/hashes and
train/validation/test years, and reconstructs states from dimensional T/u/v/ps.
Expect substantial network I/O. Do not copy an old `contract.json` into a new
run or change its hashes to bypass the implementation guard.

The physical driver also needs the prescribed Ames dust reference. Download the
original release file (large), then extract the compact surface/dust product:

```sh
rtk proxy mkdir -p data/ames/fv3betaout1
rtk proxy curl -fL https://data.nas.nasa.gov/legacygcm/fv3betaout1/fv3betaout1/03340.atmos_average_zstd.nc -o data/ames/fv3betaout1/03340.atmos_average_zstd.nc
rtk proxy .venv/bin/python scripts/stage_ames_surface_reference.py data/ames/fv3betaout1/03340.atmos_average_zstd.nc data/ames/fv3betaout1/ames_surface_reference.nc
rtk proxy env PYTHONPATH=apps/mars-calibration .venv/bin/python apps/mars-calibration/prepare_inputs.py --repo . --restart outputs/paper/windows/my32_q0_s0-restart.npz --output outputs/paper/inputs
```

Check `windows.json` and the input manifest before running. The fresh explicit
restart avoids the input freezer's legacy historical restart default. The
selected MY32 validation restart is used for gradient/ablation pilots; fitting
uses the distinct window states and targets in `windows.json`. The freezer's
daily `target.nc` is retained for the legacy driver and is not the native fit target.

## 5. Run the experiments needed for revised physical claims

First validate the manifests and run a small gradient pilot:

```sh
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py calibration --inputs outputs/paper/inputs --manifest outputs/paper/windows/windows.json --output outputs/paper/preflight --validate-only
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py gradients --inputs outputs/paper/inputs --output outputs/paper/gradient-pilot --steps 1 --epsilons 1e-3 1e-4
```

Then run the full physical derivatives, multi-diagnostic derivatives, bounded
reanalysis calibration, neural pilot, and short paired ablations. Add
`--require-gpu` on the GPU host to reject CPU fallback:

```sh
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py gradients --inputs outputs/paper/inputs --output outputs/paper/gradients --steps 1 4 16 32 74 148 --epsilons 1e-3 1e-4 1e-5
rtk proxy .venv/bin/python scripts/run_scalar_sensitivity.py --inputs outputs/paper/inputs --output outputs/paper/scalar-sensitivity --steps 74 296 1480 --dt 300 --epsilon 1e-3
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py calibration --inputs outputs/paper/inputs --manifest outputs/paper/windows/windows.json --output outputs/paper/calibration --optimizer lbfgsb --starts 3 --updates 500 --max-evaluations 1000 --gtol 1e-6 --ftol 1e-12 --seed 0
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py neural --inputs outputs/paper/inputs --manifest outputs/paper/windows/windows.json --output outputs/paper/neural-seed0 --updates 50 --seed 0
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py ablations --inputs outputs/paper/inputs --manifest outputs/paper/windows/windows.json --output outputs/paper/short-ablations --cases no_pbl no_convection no_regolith half_timestep
```

Inspect `acceptance.json`, `optimization.json`, `uncertainty.json`, `training.csv`, `skill.csv`, and
checkpoint/source snapshots. A budget/relative-loss stop is not stationarity;
report the validation-selected controls and termination for every start. Repeat
the fit under additional seeds and retain held-out year/block uncertainty. The
native 25-step target interval is 295.91748 s per step; do not force it to 300 s.

For the synthetic three-control recovery, timing and long ablation protocols:

```sh
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py --inputs outputs/paper/inputs --config apps/mars-calibration/paper-experiment.json --output outputs/paper/synthetic --task recovery
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py --inputs outputs/paper/inputs --config apps/mars-calibration/paper-experiment.json --output outputs/paper/timing --task benchmark
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py --inputs outputs/paper/inputs --config apps/mars-calibration/paper-ablations-year.json --output outputs/paper/annual --task ablations
```

Timing reports must identify actual hardware and distinguish compile/warmup from
synchronized repetitions. No matched Ames/LMD speedup is established by these runs.
The long ablation config is a **300/150 s** experiment, not 450/300 s. Do not
weaken the calibration driver's 300 s guard to recreate the v1 headline.

For fresh seasonal spinup with untuned (1,1,1) controls:

```sh
rtk proxy .venv/bin/python scripts/run_gcm3d_ablation.py data/tes/mgs_tes_surface_1deg.nc --output-dir outputs/paper/climate --config convection --truncation T21 --layers 12 --dt 300 --initial-ls 0 --diurnal --sols 2006 --chunk-sols 5 --cooldown-seconds 0 --co2-lw-scale 1 --dust-lw-scale 1 --surface-exchange-multiplier 1 --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc
rtk proxy .venv/bin/python scripts/evaluate_seasonal_convergence.py outputs/paper/climate/convection/checkpoint_diagnostics.csv --start-sol 668 --output outputs/paper/climate/convergence.json
```

2006 sols is a requested pilot, not proof of equilibration. Extend using the
identical command plus `--resume` if seasonal pressure, frost, surface T or deep
soil fail the convergence criteria. `export_mars_seasonal_checkpoints.py` exports
matched seasonal evaluation windows; `benchmark_ames.py`, `benchmark_mcd.py`,
`evaluate_arco_seasonal.py`, and `tform mars compare` perform the documented
reference comparisons. Their exact options and contracts are in
`docs/differentiable-experiments.md`, `docs/ideas/iclr-minimal-validation.md`, and
`iclr-results/01-mola-topography-comparisons/config.json`. The historical config
points to **historical sparse checkpoints**; create a new config for new results,
matching local times, output durations, wind heights, years, dust and averaging.
Do not silently use it as an independently validated new-model comparison.

## What must happen before stronger claims or issue closure

| Issue | Remaining acceptance gate / paper action |
|---|---|
| #91 / #92 | Archive a converged current-physics multi-start reanalysis fit and an independent seasonal evaluation; replace historical skill only after that run. |
| #93 | Corrected table includes bias/frost/wind and Ls225; keep sampling/tuning qualifications beside it. |
| #95 | A real common-state/common-physical-time 450/300/finer study is still absent. The historical 300/150 pair does not close it. |
| #62 / #103 | Full default T170 stability, spectra, local wind/Mach/CFL and angular-momentum sensitivity remain required. |
| #63 / #77 | Atmospheric condensation and general mixture partial-pressure/restart support are unimplemented. Keep the surface-only/fixed-gas limitation beside cycle claims. |
| #76 | Raw pre-projection mass, energy and momentum budgets plus regional correction effects are required. `diagnose_mars_energy_budget.py` is a snapshot flux diagnosis, not that closure study. |
| #74 | Trace all legacy gas coefficients to primary sources/domains or replace them with validated optics; corrected mole arithmetic alone is insufficient. |
| #102 | Publish raw headline fields, spinup, exact run revisions/patches, inputs and immutable locations. A hash or a local path does not publish a missing file. Reconcile year9/10 definitions explicitly. |

`evaluate_seasonal_field_convergence.py` compares consecutive-year spatial fields;
`analyze_differentiable_experiments.py` analyzes climate/observations/scaling;
`iclr-results/figures/make_differentiable_figures.py` and `make_figures.py` produce
diagnostic figures from archived CSV/JSON, not new simulations. Never use a
global-mean difference as though it were spatial RMS, or a constrained inventory
residual as raw conservation. Missing primary data and unimplemented physics
remain tracked work, not steps that can honestly be replaced with a command.

After new runs pass, update the claim map and frozen evidence, regenerate the
tables/figures, compile the PDF, and review the prose and issue dispositions
together. The manuscript PR intentionally retains historical provenance and
withdraws unsupported claims rather than inventing corrected annual results.
