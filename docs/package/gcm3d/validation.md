# Validation status and limitations

Software verification, numerical stability, seasonal equilibrium and independent
observational agreement are separate requirements.

## Software verification

At revision `198433d`, local package/CLI/calibration validation produced **411
passed, three skipped, no failures**, including all 11 slow tests. The UI build
passed. This records that revision rather than promising results for later code.

```bash
rtk proxy .venv/bin/python -m pytest -c package/pyproject.toml \
  package/tests cli/tests apps/mars-calibration/tests -m 'not slow' -q -rs
rtk proxy .venv/bin/python -m pytest -c package/pyproject.toml \
  package/tests cli/tests -m slow -q -rs
```

Asset-dependent tests can skip when data are absent. The explicit configuration
registers the `slow` marker across suites. Slow CLI tests include a four-year
intervention with GCM diagnostic snapshots and can take several minutes on CPU.
Tests cover dry-dycore benchmarks, orbit, energy exchange, radiation closure,
CO₂ inventory/escape, positivity, restart, gradients and experiment integrity.

## Climate equilibrium

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
- Dry convection and simplified surface/PBL closures; no water/cloud cycle.
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
