# Paper figures

Beautified, style-consistent figures for the ICLR submission, generated from the
curated archive under `iclr-results/`. Every figure is written to
`iclr-results/figures/out/` as both **PNG** (preview) and **PDF** (vector, for
`\includegraphics`).

- `plotstyle.py` — shared rcParams, colorblind-safe palette, stable
  per-reference / per-method colors, PNG+PDF export helper.
- `make_figures.py` — figure generators (reads the archive; writes nothing else).
- `render.sh` — one-shot regeneration wrapper.

## Regenerate

Run from the repository root:

```bash
# everything
bash iclr-results/figures/render.sh

# or select groups
.venv/bin/python iclr-results/figures/make_figures.py ablations year benchmarks timing neural

# custom locations
.venv/bin/python iclr-results/figures/make_figures.py --root iclr-results --out /tmp/figs all
```

Dependencies: `matplotlib`, `numpy`, `pandas` (already in `.venv`).

## Figure catalogue

| File | Group | Source | Paper slot |
|---|---|---|---|
| `ablations_rmse.{png,pdf}` | ablations | `02-ablations/short-paper-suite/ablations.csv` | Physics sensitivity (temperature + wind) |
| `ablations_conservation.{png,pdf}` | ablations | same | Energy / angular-momentum change |
| `year_ablations.{png,pdf}` | year | `02-ablations/mars-year-gpu{1,2,3}/ablations.csv` | One-Mars-year stability sensitivity |
| `benchmark_rmse_by_season.{png,pdf}` | benchmarks | `01-mola-topography-comparisons/metrics.csv` | Model vs Ames/MCD/ARCO skill vs season |
| `benchmark_correlation.{png,pdf}` | benchmarks | same | Ls-averaged pattern-correlation heatmap |
| `performance_timing.{png,pdf}` | timing | `03-benchmarks/performance-timing/timing.json` | Throughput + warm step time |
| `neural_radiation_training.{png,pdf}` | neural | `04-neural-experiments/radiation/report.json` | Training convergence (3 objectives) |
| `neural_radiation_generalization.{png,pdf}` | neural | same | Held-out (test/extrapolation) air-RMSE |
| `neural_control_descent.{png,pdf}` | neural | `04-neural-experiments/control/report.json` | Inverse-control gradient descent |
| `parameter_recovery.{png,pdf}` | neural | `parameter-recovery.csv` (GPU fallback) | Inverse parameter-recovery accuracy |

Every figure carries a small footer citing its source and the standing caveat
(diagnostic/transient run; RMSE is relative to each run's `full` case, not
observational truth).

## Style conventions (keep consistent if you add figures)

- Import `plotstyle` and call `apply_style()` before plotting.
- Use `PALETTE` / `REFERENCE_COLORS` / `METHOD_COLORS` so a reference or method
  keeps the same color in every panel.
- Save with `plotstyle.save(fig, out, "stem")` (PNG+PDF, 300 dpi, tight bbox).
- Add a one-line `annotate_provenance(fig, ...)` footer.

---

# Regenerating the underlying results

The figures only read curated artifacts. To regenerate the *results* themselves
(GPU recommended for the paper-scale runs), use the recorded invocations.

### Physics ablations (short suite + one Mars year)

```bash
# Short paper ablation suite  -> outputs/paper-ablations-gpu
.venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/paper-ablations-gpu \
  --task ablations --require-gpu

# One-Mars-year partitions (split across GPUs to fit wall-clock)
.venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-ablations-year.json \
  --output outputs/paper-year-gpu1 --task ablations --require-gpu \
  --cases no_co2_exchange weaker_diffusion surface_exchange_multiplier_0.8
# ...repeat with paper-year-gpu2 / gpu3 for the remaining cases
```

### Reference comparison + performance timing

```bash
# Ames / MCD / ARCO seasonal comparison -> outputs/reference-comparison
.venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/reference-comparison --task recovery --require-gpu

# Throughput / wall-clock timing -> outputs/paper-timing-gpu
.venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/paper-timing-gpu --task benchmark --require-gpu
```

### Neural framework experiments (CPU is sufficient for defaults)

```bash
# Physical parameter recovery -> outputs/neural_framework/recovery
env JAX_PLATFORMS=cpu .venv/bin/python examples/neural/recover_parameter.py \
  --iterations 120 --steps 24 --output outputs/neural_framework/recovery

# Neural radiation (local / trajectory / continued-local)
env JAX_PLATFORMS=cpu .venv/bin/python examples/neural/train_radiation.py \
  --epochs 100 --fine-tune-epochs 20 --columns 96 --layers 4 \
  --steps 12 --long-steps 48 --coupled-steps 2 \
  --output outputs/neural_framework/radiation

# Inverse atmospheric control
env JAX_PLATFORMS=cpu .venv/bin/python examples/neural/optimize_atmospheric_control.py \
  --iterations 12 --steps 3 --layers 2 \
  --output outputs/neural_framework/control_final_smoke
```

After regenerating, re-copy the curated artifacts into `iclr-results/` and rerun
`render.sh`.

---

# Suggested results that would strengthen the paper

Ordered by expected reviewer impact vs. effort. Items marked **(claim gap)**
address a limitation the current archive explicitly flags.

### High impact
1. **Equilibrated multi-year climatology (claim gap).** Every current run is a
   transient from a restart. A >=2–3 Mars-year run showing a repeating seasonal
   CO2/pressure cycle (Viking-lander-style pressure curve) converts "numerically
   completes" into "reproduces the observed annual cycle". *Figure:* surface
   pressure vs Ls overlaid on Viking Lander 1/2. `run_gcm3d_ablation.py --sols
   ~1400`.
2. **Held-out observational validation, not model-vs-model.** The reference
   comparison is labeled "not observational truth." Adding TES/MCS assimilated
   temperature or MOLA-derived surface pressure as an independent target — even
   for one season — is the single strongest credibility upgrade. Reuse
   `compare_mars_reference.py` with an observational target.
3. **Ablation significance / uncertainty.** Repeat the short ablation suite over
   >=3 seeds (or initial `Ls`) and plot mean ± spread so RMSE differences are
   distinguishable from run-to-run noise. Turns the bar charts into an
   error-barred, defensible sensitivity table.

### Medium impact
4. **Gradient-based calibration end-to-end.** You already show parameter
   recovery on generated data; a short demo recovering a physical parameter
   against a *reference* dataset (not self-generated) demonstrates the
   differentiability payoff on real data. *Figure:* loss + parameter trajectory.
5. **Neural-corrected vs baseline forecast skill.** The neural radiation result
   shows fit quality but "no forecast skill." Report RMSE of a
   physics+neural-tendency rollout vs pure physics on a held-out initial state.
   Even a small, honest improvement (or a clean null) is publishable and
   directly tied to the framework's thesis.
6. **Resolution / vertical-layer scaling.** Rerun the reference comparison at
   T21/T42 and 12/20 layers; plot skill and throughput vs resolution. Pairs the
   `performance_timing` figure with an accuracy axis (accuracy–cost frontier).
7. **Gradient-accuracy check.** Finite-difference vs autodiff gradient agreement
   across step counts / horizons (extend the recovery `gradient_checks` block).
   *Figure:* relative gradient error vs rollout length — validates the
   differentiable core the paper depends on.

### Lower effort, still valuable
8. **Timestep / conservation convergence.** You already have `half_timestep`;
   add 2–3 more dt values and plot RMSE-vs-dt (order of accuracy) and
   energy/AAM drift-vs-dt. Strengthens the numerics claim.
9. **Wall-clock vs external simulators.** `timing.json` lists Ames and LMD/MCD as
   `not_measured`. Even a single documented reference number would let you make a
   bounded, defensible efficiency statement instead of none.
10. **Qualitative map panels.** A 2x2 of modeled vs reference surface
    temperature/pressure fields at one `Ls` (from the existing `*_grid.png` /
    `*_differences.png`) gives reviewers a visual sanity check alongside the
    quantitative RMSE curves.

> Keep the standing caveats in every new figure caption: transient vs
> equilibrated, model-vs-model vs observational, generated-data vs Mars skill.
