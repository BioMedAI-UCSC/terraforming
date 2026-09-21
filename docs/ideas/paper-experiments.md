# Reproducible experiments for the physical Mars GCM paper

The new runner addresses trajectory-based inference, controlled process and
parameter ablations, and forward/derivative timing. It can rebuild the existing
cached reference comparison. It does not train a neural component or claim
external simulator speedups without measurements.

## Protocol

The frozen [paper configuration](../../apps/mars-calibration/paper-experiment.json)
uses T21/L12, 300-second steps, all current physical operators, diurnal forcing,
MOLA, TES and prescribed Ames seasonal dust. It reads the hashed existing
`outputs/nautilus-calibration-inputs` bundle. The ARCO `target.nc` is integrity
checked for provenance but is **not used as a synthetic training target**.

The teacher parameters are `(0.6, 0.8, 1.4)` for CO₂ opacity, dust opacity and
surface exchange. Training observations occur at exactly 6,000, 12,000 and
22,200 seconds after the fixed restart. The excluded continuation uses 30,000
and 44,400 seconds. Targets include all atmospheric temperature and horizontal
wind levels. Synthetic targets and fitted trajectories are saved as NPZ arrays
in `(time, field, layer, longitude, latitude)` order, with fields `(T,u,v)`.

Fit surface exchange alone first, keeping other parameters at known truth;
then fit all three parameters. Three declared initial guesses are used.
Normalization is calculated only from training timestamps, with standard
deviation floors of 1 K or 1 m/s. Loss uses Gaussian horizontal weights and
equal layer, time and field weights. Later observations do not enter fitting.

Both optimizers use the same normalized bounded parameter coordinates and a
100-oracle-call **ceiling**, stopping early if converged. Unlike the frozen
Phase-1 protocol, these runs do not pad convergence with optimizer restarts.
L-BFGS-B obtains loss and forward-mode derivatives together; Powell uses forward
losses. Actual calls, optimizer overhead, wall time, compilation and warmup are
reported separately. This budget gives Powell more opportunity to explore all
coordinates than the old 20-call experiment; inspect the traces for actual
coverage. Equal calls are not equal compute.

At each initial guess, centered finite differences at `1e-3` and `1e-4` test
every active derivative. Acceptance requires absolute gradient error at most
`1e-10 + 1e-4 * abs(finite_difference)`, recovery within 5% for every active
parameter, and reduced training and continuation loss. Each optimizer/start
receives its own verdict. Low loss alone is not parameter recovery. The overall
recovery verdict conservatively requires every requested case to pass.

This is a noise-free identical-model experiment and can expose identifiability
problems. It does not establish real-data calibration, independent-year skill,
or robustness to observation noise. Initial soil/frost are held exactly fixed.

## Run and verify

Use a new output directory for every invocation; existing output directories
are refused. No data downloads or cluster submissions occur automatically.

```sh
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/paper-recovery --validate-only

rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-smoke.json \
  --output outputs/paper-smoke --task all

rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/paper-recovery --task recovery --require-gpu
```

The smoke profile has two training samples and two later samples over four
physical steps, one starting guess, one active parameter, and eight oracle
calls. It tests execution; it is not a paper result or guaranteed recovery.
`--require-gpu` fails instead of falling back to CPU. Exit 10 means execution
completed but a requested recovery criterion failed. Ordinary execution errors
produce `failure.json` and a nonzero exit. Read `report.json` to distinguish
scientific acceptance from completion. Each experiment writes live JSONL progress.

The existing Phase-1 command and `experiment.json` remain separate. New installed
entry point: `mars-paper-experiments` (same options as the script).

## Controlled ablations and numerical sensitivity

```sh
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-experiment.json \
  --output outputs/paper-ablations --task ablations
```

The 13 cases are full physics; no PBL; no convection; no regolith conduction;
no CO₂ exchange; weaker diffusion (0.2-sol decay time versus 0.1); half timestep;
and ±20% sweeps of each of the three parameters. Each starts from the same restart
and samples at the same physical times. `--cases no_pbl half_timestep` selects a
subset, always including full physics. The baseline uses the declared teacher
parameters; it is not silently selected from successful recovery trials.

Removal is paired: no regolith removes the surface and soil conductive exchange;
no CO₂ exchange removes pressure, frost and latent tendencies together, freezes
frost, and retains the global inventory projection. Diffusion times stay in
physical units when timestep changes. The existing PBL closure interval and CO₂
supply limiter interval stay fixed, isolating the outer timestep. Only IMEX-RK-SIL3 is used; timestep
sensitivity is not advertised as an integrator comparison.

The runner exports sampled full-state NetCDF, a final restart, extrema, global
CO₂ drift, atmospheric energy and angular-momentum changes, paired temperature/
wind RMSE versus full physics, and runtime. Non-finite ablations are recorded
as failures rather than dropped; a non-finite baseline aborts the experiment.
These are **transient sensitivities**, not equilibrium or independent accuracy
scores. Energy excludes surface geopotential and these changes are **not closed
forced energy/torque budgets**. Longer seasons require a separate declared
configuration and adequate runtime; the default short window cannot establish
that slow soil/frost processes are unimportant.

## Reference comparisons and speed

Add `--reference-config cli/configs/reference-comparison.json` to rebuild the
existing four-season reference report. Its input hashes and sampling limitations
are preserved. Its model paths still refer to the historical nominal baseline;
this does not automatically compare the new fitted parameters. The new ablation
NetCDF files use the existing full-state schema so that subsequent matching
pipelines can consume them. Do not substitute short samples for seasonal means.

`--task benchmark` measures repeated synchronized forward trajectories and loss
plus three forward tangents on the same hardware. Compilation and warmup are
separate, and persistent compilation-cache effects are disclosed. Compiled
temporary memory is reported separately from whole-process peak host RSS;
unmeasured peak accelerator memory remains null.

Optional `--external-timings external.json` imports a nonempty list of records
with these fields: `simulator`, `source` (citation or measurement provenance),
`hardware`, `resolution`, `physics`, `precision`, `timed_region`,
`simulated_seconds`, `wall_seconds`, and `dt_seconds`. Numeric fields must be
positive and finite. Native seconds avoid Earth-day/Mars-sol confusion. Rows
are normalized to simulated seconds per wall second but no unmatched speedup
ratio is invented. Without this file, Ames and LMD timings say `not_measured`.
MCD response time and archived output download time are not simulation time.

## Outputs and remaining scientific work

`recovery.json`, `parameter-recovery.csv`, `optimization-traces.csv`, `trajectory-errors.csv` and
`recovery.png` support the inference figures. `ablations.json`/`ablations.csv`
support the process table. `timing.json`/`timing.csv` support runtime reporting.
Configuration, input hashes, software versions, source-tree status and artifact
hashes accompany each run. Preserve complete bundles, including rejected runs.

Before claiming the complete paper result, run the full protocol, inspect
parameter identifiability, extend the paired comparisons to relevant seasons,
perform a temporally aligned external-data calibration/validation experiment,
and obtain an actual external simulator runtime measurement or a suitably
documented published one. These are not replaced by successful smoke tests.

```sh
rtk proxy .venv/bin/python -m pytest apps/mars-calibration/tests -q
```

## Verified implementation run — 2026-09-21

The corrected [smoke report](../../outputs/paper-protocol-smoke-v2/report.json)
completed recovery, all 13 ablations, timing, and the four-season cached
reference report. This uses the existing 1,800-second PBL closure interval and
24,000-second CO₂ supply-limiter interval, unchanged across outer timestep
variants. The earlier `paper-protocol-smoke` directory is marked superseded
because its development runner varied those intervals too.

| Smoke check | Measured result |
|---|---|
| Known surface-exchange multiplier | 1.4 |
| L-BFGS-B recovery | 1.3999999835, 6 calls |
| Powell recovery | 1.3999998352, 8 calls |
| Largest initial gradient relative error | 1.43813e-6 |
| Initial later-trajectory loss | 2.35954e-5 |
| Later loss after L-BFGS-B / Powell | 1.18504e-20 / 1.18272e-18 |
| Finite sampled ablation outputs | 13 of 13 |
| Maximum sampled relative CO₂ drift across cases | 3.37508e-14 |
| Artifact hashes independently checked | 79 |
| Implementation source hashes independently checked | 4 |

Direct output checks confirm that the no-CO₂ variant preserves frost exactly
and the no-regolith variant preserves soil temperatures exactly. The application
suite passes 19 tests, including false-recovery rejection and preservation of
Phase-1 closure intervals. The focused physics/legacy application run passed
46 tests during implementation. Full paper configuration and input hashes pass
preflight.

This is two training samples over two 300-second steps and two excluded later
samples through 1,200 seconds. It is an execution and recovery demonstration,
not the full three-start, three-parameter paper experiment. Both optimizers
recover the scalar; this result alone does not demonstrate a runtime advantage
for gradients. External simulator timings remain `not_measured`.
