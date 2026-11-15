# Validation status and limitations

Software verification, numerical stability, seasonal equilibrium and independent
observational agreement are separate requirements.

## Software verification

At revision `198433d`, local package/CLI/calibration validation produced **411
passed, three skipped, no failures**, including all 11 slow tests. The UI build
passed. This records that revision rather than promising results for later code.

```bash
.venv/bin/python -m pytest -c package/pyproject.toml \
  package/tests cli/tests apps/mars-calibration/tests -m 'not slow' -q -rs
.venv/bin/python -m pytest -c package/pyproject.toml \
  package/tests cli/tests -m slow -q -rs
```

Asset-dependent tests can skip when data are absent. The explicit configuration
registers the `slow` marker across suites. Slow CLI tests include a four-year
intervention with GCM diagnostic snapshots and can take several minutes on CPU.
Tests cover dry-dycore benchmarks, orbit, energy exchange, radiation closure,
CO₂ inventory/escape, positivity, restart, gradients and experiment integrity.

On November 13, the local CO₂/frost-optics changes passed **263 GCM and
calibration tests**, with three skips and three slow tests excluded. The coupled
eight-step albedo derivative check passed; its maximum finite-difference relative
error was `4.75e-9`. These are software checks of the local changes, not a new
seasonal-climate result.

The isolated PR branch passed 262 GCM/calibration tests, with three stale
deprecated-ODE skips and three slow tests excluded. Two focused cache checks
also passed, including the slow native-observation T21/L12 forecast. The
workspace-only control-optimizer test is outside this PR, which explains the
one-test count difference. Both logs are in the evidence bundle.

## Ames reference and CO₂ mechanism check

The reference is the local Ames release's
`build_run/fms_mars_default_v3.2`, `mars_surface.F90:albedo_calc` and
`update_mars_atmos.F90:co2_condense`. Its standard setup enables surface and
atmospheric CO₂ condensation and prescribed background dust. Interactive dust
and gravity-wave drag are disabled in that setup.

The JAX model now includes atmospheric phase change, transported CO₂ ice,
settling, frost surface optics and reflected infrared at the ground. Its
finite-rate cloud exchange differs from Ames's immediate deposition scheme.
It needs its own tests and climate comparisons.

The check requires the local Ames source at `AmesGCM/` so its reference files can
be saved with the executed source. Run the paired synthetic polar-night check with:

```bash
.venv/bin/python scripts/validate_mars_co2_physics.py --sols 5 --dt 300 \
  --output outputs/iclr_validation/co2-polar-night-verified-300s
.venv/bin/python scripts/validate_mars_co2_physics.py --sols 5 --dt 150 \
  --output outputs/iclr_validation/co2-polar-night-verified-150s
```

Each pair starts from the same dimensional state, 10 K below the layer frost
point. Both cases use zero sunlight, flat terrain, background dust, regolith,
PBL and dry convection. Only atmospheric CO₂ phase change and settling differ.
The report includes the full configuration, source snapshot and hashes,
initial/final restarts, per-step frost deficits, raw mass errors and projection
energy changes. This checks the cold-air mechanism and numerical behavior.
Seasonal equilibrium and observational skill require separate experiments.

The November 13 five-sol pairs passed with identical initial states and matching
executed-source hashes. Final maximum air frost-point deficits were:

| Timestep | Surface-only cycle | Atmospheric phase change and settling |
|---|---|---|
| 300 s | 13.510594 K | 0.091409 K |
| 150 s | 13.510595 K | 0.091410 K |

The reduction is 99.32% at both timesteps. The final mean-pressure difference
between condensation runs was 0.00255 Pa; the mean surface-temperature difference
was 0.000542 K. Maximum total CO₂ inventory drift **after pressure correction**
was `5.32e-13` across these runs. Raw step mass errors remain nonzero: for the
condensation case the maximum fell from `1.59e8` kg at 300 s to `1.90e7` kg at
150 s. Energy changes from pressure correction are retained in the reports.

The small raw evidence bundle is tracked in `iclr-results/06-co2-physics/`.
It includes both paired reports, initial/final states, per-step histories,
executed-source archives, a comparison figure and a verification/plotting script.
Run `python iclr-results/06-co2-physics/summarize_co2_physics.py` to check archive
hashes, confirm identical initial states and regenerate the summary and figure.
The Ames reference notice and license accompany the source snapshots.

A separate T21/L12 five-sol cold start with MOLA terrain, TES surface fields,
seasonal Ames dust and diurnal sunlight remained finite. At the saved one-sol
checkpoints, maximum Mach number was 0.569 and maximum horizontal advective CFL
was 0.348. The latter measures wind travel relative to grid-cell width during a
step; it is not a stability bound for the complete spectral integrator. The
post-correction CO₂ inventory drift was below `9.9e-14`. These short pilot results
do not certify year-long, high-resolution or equilibrium behavior.

Checkpoint diagnostics retain `atmosphere_plus_surface_reservoir_mass_kg` for gas
and ground frost. `total_co2_mass_kg` also includes suspended ice. Conservation
reports use the latter for new runs. Historical files without that field retain
their original gas-plus-surface definition. New ablation runs save executed
source and reject resumes with different source hashes or CO₂ settings.

## Issue #104 status (November 13, 2026)

The [tracker](https://github.com/BioMedAI-UCSC/terraforming/issues/104) remains open.
PRs #115 and #116 are open. PR #117 was closed without merging; its manuscript
changes have not landed through that PR.

| Requirement | Current disposition |
|---|---|
| #63 atmospheric condensation | Local implementation and paired mechanism tests; seasonal and reference validation remain. |
| #62, #95, #103 numerical stability and winds | Default damping and short pilots; T170, production 450/300/finer comparisons and wind-reference distributions remain. |
| #76 conservation | Raw and corrected exchange diagnostics include clouds; complete dry-core energy/momentum closure and regional correction studies remain. |
| #91, #92 independent evaluation and fit | Converged current-physics calibration, held-out years and uncertainty remain. |
| #93, #99, #101 reporting and neural radiation | Companion PR work has not landed on main; rerun affected learning results after the infrared-boundary change. |
| #74, #77, #97 interventions | Coefficient validation and 3-D gas-mixture physics remain; #116's separate legacy-unit fix is open. |
| #102 reproducibility | New pilots save complete local source snapshots; public headline fields and immutable external artifact publication remain. |

## Climate equilibrium

The current CO₂ cloud and frost-optics changes need new seasonal runs. Earlier
equilibrium records below belong to their original physics and input files.

The recorded corrected baseline completed two additional Mars years with finite
fields and maximum combined atmosphere/frost CO₂ relative drift about `1.81e-9`.
Phase-matched pressure, frost and surface-temperature changes passed 5 Pa, 5 Pa
and 1 K criteria. Deep-soil changes of 0.442–0.634 K exceeded 0.25 K, so the full
equilibrium gate failed. These are recorded results in the
[progress report](../../ideas/progress-report.md), not new documentation-build runs.

`MarsMapFields.is_transient` uses duration below 668 sols as a heuristic. Passing
that duration is not an equilibrium diagnostic. Longer runs still need matched
seasonal repeatability, bounded budgets and appropriate averaging.

## Reference diagnostics

Ames FV3 is an independent model reference, MCD is model-derived climatology and
ARCO-MACDA is reanalysis. Their existing seasonal comparisons show useful spatial
structure and remaining wind/frost deficiencies. Sampling, dust scenarios, wind
heights and frost definitions differ. See [reference comparisons](../../cli/reference-comparison.md).
These diagnostics do not establish independent Viking/TES/MCS observational skill.

The saved ablation audit verifies output integrity and paired protocols but records
a physics source-hash mismatch. Saved-run results belong to their original source
hashes; they do not validate a later code revision merely because they are committed.

## Remaining limitations

- Prescribed dust opacity; no lifting, transport or interactive dust weather.
- Dry convection and simplified surface/PBL closures; no water cycle or cloud radiation.
- CO₂ clouds use a fixed particle size and share gas temperature and horizontal
  velocity. Phase change assumes CO₂ pressure equals total gas pressure.
- Homogeneous regolith properties and zero-flux bottom boundary; slow soil spin-up.
- No photochemistry or composition-dependent high-pressure validation.
- Global CO₂ projection does not prove local transport accuracy or conservation of
  every energy/momentum term.
- Bounded neural heating can add net energy; it is not a closed energy source.
- GCM intervention snapshots are independent spin-ups from a torch global-mean
  trajectory, not a continuous multi-year 3-D intervention integration.
- Higher resolution presets do not themselves establish convergence or skill.

The [physics ledger](../../ideas/gcm3d-physics-limitations.md) retains detailed
requirements and references. Date-specific design notes are historical context;
this section describes the current implementation.

The torch global-mean model is [deprecated](../../deprecated-global-mean.md).
Its remaining intervention dependency is documented for compatibility; use GCM
workflows for new spatial-model research.
