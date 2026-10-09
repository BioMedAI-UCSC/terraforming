# One-Martian-year ablations

Use `apps/mars-calibration/paper-ablations-year.json` only with `--task ablations`.
This extends the existing paired experiment without changing its initial state,
parameters, physics, closure intervals, or integrator. The duration is exactly
59,356,800 seconds, the model's orbital period: 197,856 steps at 300 seconds,
or 395,712 for the half-timestep case. This is a Mars year, not an Earth year.
Solar forcing and the prescribed seasonal dust climatology advance with model
time. This is not a reconstruction of observed dust weather in a specific year.

The configuration retains the five original short-run samples, adds early
samples, then saves approximately every ten sols and at the year endpoint.
All sample offsets are exact multiples of both timesteps. `train_seconds` and
`validation_seconds` are concatenated for ablations; no fitting is performed.
The combined RMSE is a sample average, not a time-weighted annual average.
Saved snapshots are instantaneous, not daily or seasonal means.

## First run the full-physics baseline

```bash
mamba run python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-ablations-year.json \
  --output outputs/paper-year-baseline-gpu \
  --task ablations --cases full --require-gpu
```

Verify finite fields, temperature and wind histories, pressure/frost exchange,
and atmospheric-plus-frost mass drift before spending time on all ablations.
Baseline RMSE against itself is zero by construction and is not a validation.

## Then run all 13 cases

```bash
mamba run python scripts/run_mars_paper_experiments.py \
  --inputs outputs/nautilus-calibration-inputs \
  --config apps/mars-calibration/paper-ablations-year.json \
  --output outputs/paper-ablations-year-gpu \
  --task ablations --require-gpu
```

Use new output directories; existing directories are rejected. Run on a durable
GPU job allocation. The runner executes each complete trajectory twice (warmup
and measurement), and saves outputs only after each case completes. It has no
mid-case checkpoint/resume or early-stop monitoring. A nonfinite baseline aborts
the suite; nonfinite ablations are recorded. Sparse outputs cannot certify
physical bounds at every intermediate step.

Linear extrapolation from the measured half-sol GPU runs suggests about three
hours for baseline warmup plus measurement, and roughly 44 hours for the full
suite, excluding compilation and I/O. These are estimates, not benchmarks;
hardware, memory pressure, and longer trajectories may change the cost. The
full suite reruns the baseline; it does not reuse the first command's results.

## Interpretation

This tests a one-year transient from the existing restart, not an equilibrated
climate. Check time-resolved diagnostics as well as aggregate RMSE. Long-horizon
chaotic divergence can increase timestep differences without proving numerical
failure; compare seasonal statistics separately before claiming convergence.
No acceptance threshold for long-run physical realism is established by this
configuration. Keep any unstable ablations in the report rather than omitting
them. Energy and angular momentum changes are not closed forced budgets.

Success supports year-long forward integration and seasonal physics sensitivity.
It does not establish observational accuracy, independent-year generalization,
equilibrium, or year-long differentiability. Existing gradient/recovery evidence
remains short-horizon. No alternative integrator or neural model is tested.
