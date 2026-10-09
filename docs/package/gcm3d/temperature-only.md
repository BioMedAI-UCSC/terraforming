# Frozen temperature-only postprocessing

This workflow fits a correction to cached physical forecasts. Unlike a
[neural tendency inside the solver](neural-experiments.md), fitting does not
differentiate through GCM integration or change wind, pressure, frost or soil.

The native ARCO-MACDA contract selects 112 training, 32 validation and 32 test
starts with fixed seasonal coverage, observed initial fields, past-only soil
initialization and exact native time alignment. Statistics and controls are
fitted on training data only.

## Workflow and acceptance

1. Stage windows and create the frozen manifest/contract on CPU.
2. Cache 144 training/validation forecasts on selected GPUs.
3. Release GPUs, fit constant/ridge/neural controls on CPU, and freeze models.
4. Cache 32 test forecasts only after passing the validation gate.
5. Generate and verify reports, predictions, model hashes and budget records.

Follow the [launch and acceptance runbook](../../neural-temp-only-runbook.md)
for complete commands, GPU requirements, failure and resume behavior.
The cache launcher rejects CPU fallback; synthetic tests, staging and fitting
can run on CPU.

Validation requires at least 10% RMSE reduction against physical forecasts and
better RMSE than ridge. Test acceptance additionally checks the no-GCM control
and a positive paired temporal-block interval for linear-minus-neural MSE.
Bootstrap units are whole Mars-year/season blocks, not grid cells. The fixed
budget is four aggregate GPU-hours, not four hours on each GPU.

Inspect `scientific_success`, `test_evaluated`, coverage and checksums. A stopped
validation gate is a valid completed negative experiment. Synthetic tests do not
establish held-out Mars skill; the full 176-start remote campaign is not validated
by local tests.

## Reproducibility

Keep native revision, source hashes, dependencies, terrain checksum, feature schema
and physical settings with the run. Scientific changes need a new run directory.
Completed forecasts are reused only if their contract matches.

This pipeline's trusted local pickle training checkpoints differ from the generic
neural framework's pickle-free inference NPZ files. Do not load untrusted pickle
artifacts. See the [implementation recap](../../neural-temp-only-implementation-recap.md).
