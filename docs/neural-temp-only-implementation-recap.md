# Temperature-only neural implementation recap

Date: 2026-09-23  
Branch: `neural-temp-only`  
Implementation commit: `327c580`  
Remote: `origin` (`BioMedAI-UCSC/terraforming`)

The temperature-only plan is implemented and locally verified. The bounded
176-start remote experiment has **not** been run, so no held-out Mars forecast
improvement is claimed. The [runbook](neural-temp-only-runbook.md) contains the
complete launch sequence and expected results.

## Delivered

| Component | Implementation | Behavior |
|---|---|---|
| Native data and contracts | `scripts/neural_temp_data.py` | Pins archive revision; scans actual time/year metadata; selects disjoint seasonal windows; validates cadence, units, fields and splits; requires 112/32/32 coverage; freezes provenance, source indices, checksums, software and scientific configuration. |
| Physical forecast cache | `scripts/cache_neural_temp.py` | Initializes observed atmosphere and start-only surface/frost; spins soil from past observations; runs frozen T21/L12 float64 physics over MOLA for exactly 1/12 sol; fixes start dust; writes atomic, checked entries with labels separate from inference features. |
| Features and residual network | `scripts/neural_temp_model.py` | Fixed 24-channel schema; training-only normalization and residual RMS; shared 32/32 GELU network with zero output initialization; no feedback into the physical solver. |
| Fitting and controls | `scripts/train_neural_temp.py` | Physical, persistence, constant, ridge, neural and no-GCM controls; prescribed optimizer and sampler; validation selection including the zero checkpoint; deterministic epoch-boundary restart; immutable frozen models. |
| Evaluation | `scripts/evaluate_neural_temp.py` | Equal-start area-weighted metrics, secondary pressure weights, all prescribed breakdowns, paired temporal-block bootstrap, prediction arrays, diagnostic plots, Markdown/JSON reports and sealed test results. |
| Bounded launch | `scripts/run_neural_temp.py` | Four distinct masked GPU workers with placement checks; compilation-inclusive pilot and cost projection; persistent aggregate budgets; timeout/failure propagation; restartable cache; validation-gated test; artifact verification. |
| Verification | `package/tests/gcm3d/test_neural_temp.py` | Data leakage, metrics, model learning, restart equivalence, cache integrity, launcher failures, budget/recovery behavior, full synthetic CLI workflow and optional real physical smoke test. |

The existing daily-mean MACDA loader and physical equations remain unchanged.
The new pipeline reuses their spatial regridding, spectral projection and
physical solver conventions.

## Scientific details fixed during implementation

- MACDA's epoch is midnight at longitude zero; the solver's zero clock is local
  noon. A half-sol clock offset aligns them. The eccentric orbit is anchored to
  each start's solar longitude; seasons are not advanced with a uniform angle.
- Atmospheric temperature means the lowest sigma midpoint, approximately
  0.9583. It does not mean surface or measured 2-m temperature.
- The one-native-interval forecast uses 25 equal steps of approximately
  295.91748 seconds. PBL diffusion, regolith, stability exchange, convective
  adjustment, radiation and the positivity-preserving CO2 solver are enabled.
- Soil starts uniformly at the oldest historical surface temperature and uses
  the preceding sol's piecewise-constant surface boundary with backward-Euler
  finite-volume conduction and an insulated bottom.
- The no-GCM control removes every forecast channel **and** the physical
  additive temperature. It predicts change from start temperature with its own
  training residual scale.
- Metadata staging, neural fitting and reporting are separate CPU commands.
  GPU cache stages exit so the four-GPU allocation can be released between them.
  External scheduler/cloud reservation time must also respect the overall cap.
- The global concatenated archive attribute does not establish per-window
  observing-system provenance; unknown provenance is explicitly recorded.

## Verification completed

These commands passed locally:

```bash
rtk proxy .venv/bin/python -m pytest package/tests/gcm3d/test_neural_temp.py -q
```

Result: **19 passed**. This includes a real native-observation T21/L12/MOLA
forecast smoke test producing finite fields and geographic day/night coverage.
It is an execution check, not a held-out forecast skill result.

```bash
rtk proxy .venv/bin/python -m pytest package/tests/gcm3d/test_physics.py package/tests/gcm3d/test_orbit.py package/tests/gcm3d/test_topography.py -m 'not slow' -q
```

Result: **59 passed**. Together, **78 checks passed**. The six scripts also
parsed successfully, the runner's command-line help worked, and the staged
diff passed `git diff --cached --check`.

The synthetic integration test exercises all 176 cache entries, both neural
fits, all six controls, validation freezing, the test gate, final evaluation,
plots, report checksums and repeated-command reuse. A synthetic gain establishes
optimizer and pipeline behavior only. Future-observation perturbation checks
confirm that target changes do not alter inference features. Interrupted and
uninterrupted training agree exactly at the checked epoch boundary.

Pinned archive metadata was also read to confirm the source units, its special
`calendar=none` time encoding, and the documented midnight epoch.

## Remaining execution and acceptance

Follow the [runbook](neural-temp-only-runbook.md) for environment preparation,
CPU native staging, the train/validation GPU cache stage, CPU fitting, and the
conditional test GPU cache stage. Run from the repository root.

The initial real-data expectation is exactly **112 training, 32 validation and
32 test starts**. Coverage deficits, invalid inputs, over-budget projections,
worker failures and timeouts are explicit stops, not successful reduced runs.

The validation gate requires at least 10% neural RMSE reduction versus the
physical baseline and lower RMSE than ridge. If it fails, stop without test
caching and finalize a report with `scientific_success: false`.

After a passing gate and completed test cache, finalize and verify:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --stage report
rtk proxy .venv/bin/python scripts/run_neural_temp.py --run-dir outputs/neural_temp/temp_only_v1 --verify
rtk read outputs/neural_temp/temp_only_v1/report.md
```

Expected for a completed held-out experiment: intact artifacts, `status:
complete`, and `test_evaluated: true`. Scientific success additionally requires:

1. At least **10%** relative primary test RMSE reduction versus physical.
2. Lower RMSE than fitted ridge, with the **95% paired temporal-block interval
   for MSE improvement entirely above zero**.
3. Lower RMSE than the matched **no-GCM** model.
4. Total aggregate GPU time within the **four-hour** cap.

Use the machine-checkable acceptance command after artifact verification:

```bash
rtk proxy .venv/bin/python -c 'import json; r=json.load(open("outputs/neural_temp/temp_only_v1/result.json")); assert r["status"] == "complete" and r["test_evaluated"] and r["scientific_success"] and r["gpu_hours"] <= 4, r; print("PASS: fixed held-out criteria and GPU budget")'
```

Expected **only if scientific criteria are met**:
`PASS: fixed held-out criteria and GPU budget`. A failed validation gate or test
criterion must fail this command. The primary metric, layer, lead, geographic
coverage and test cases must not be changed after inspecting the results.

The experiment remains limited to one dataset and seed. Even a successful
temperature postprocessor would not establish improved physical mixing,
climate stability, coupled correction or benefits from GCM differentiation.
