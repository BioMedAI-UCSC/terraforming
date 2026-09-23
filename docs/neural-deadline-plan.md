# Neural result: two-day experiment budget

The implementation now supports a [four-method concurrent GPU screen](neural-screen.md)
with shorter defaults, table logs, and an actual equal-weight forecast ensemble.
Use that launcher for the requested side-by-side comparison; the standalone
commands below remain available for individual follow-up runs.

The deliverable requires a neural component. Do not extend the current PBL run
to 4000 updates solely because its training loss is finite. Its batches differ,
so batch loss cannot establish a plateau; compare the fixed validation baseline
and checkpoint instead. The remote checkpoint's validation results are still
needed before judging its latest checkpoint.

Received evidence (2026-09-23): the remote step-634 monitoring report has physical
loss 5.1582397533 and neural loss 5.1441295483, a 0.27355% improvement on four cases.
It does not measure the latest saved step 1622 or establish a learning plateau.
The local constant-screen smoke completed ten forecasts on one cached validation
case in about 40 seconds on CPU, with no optimization. The best constant improved
global loss by 0.07047%. That smoke uses a step-3 checkpoint and zero atmospheric
spinup, so its score is not directly comparable with the remote run. It verifies
the screening tool and supplies only a narrow sensitivity observation.

## Primary experiment: small hybrid

Use a new experiment directory and the observed-atmosphere hybrid. Its learned
temperature/wind tendencies have broader influence than the two bounded PBL
multipliers. It also avoids the old trainer's free-running atmospheric warmup;
only soil is initialized from past observations. Production-horizon GPU cost and
real-data learning remain unverified. Do not launch the width-384/five-block,
4000-step curriculum as the next experiment.

From the repository root in the activated GPU environment, with MOLA_PATH set:

```bash
timeout --signal=INT --kill-after=60s 1h bash scripts/launch_neural_hybrid_4gpu.sh \
  --output outputs/neural_hybrid/deadline_full \
  --width 64 --blocks 2 --learning-rate 0.001 \
  --chunks 4 --normalization-chunks 4 \
  --curriculum 1:100 --steps 25 --validation-horizon 1 \
  --validation-examples 4 --validate-every-steps 25 \
  --validate-every-epochs 0
```

This Linux command uses GNU `timeout` to cap wall time as well as optimizer
updates. This is a pilot, not a claim that four adjacent training windows establish
generalization. The one-hour allocation includes compilation and validation.
Checkpoints are saved each completed update. If it exceeds that
allocation, preserve the checkpoint and diagnose timing before extending it.

Read `small_validation.json` and `validation.jsonl`, especially `neural_loss`,
`physical_loss`, `latent_loss`, paired improvements, and per-field/layer RMSE.
At step 25, assess numerical stability and runtime, not convergence. If affordable
and finite, repeat the identical command with `--steps 100 --resume`. The
curriculum must remain `1:100` and all architecture/data arguments must match.

At step 100, a practical prioritization target is at least 1% relative improvement
over the physical baseline on the fixed validation sample, with improvements in
multiple cases. This is an operational budget rule, not a significance threshold
or expected result. Check the per-layer metrics; a reduced aggregate must not
conceal a serious regression in a field. For a learned-dynamics claim, internal
loss must also improve relative to the physical baseline. Stop extensions if
there is no useful validation improvement; do not infer that every neural method
has failed.

## Required inexpensive comparison

Run the same pilot with `--ablation adapters-only`, in a separate directory such
as `outputs/neural_hybrid/deadline_adapters`. Keep initialization, normalization,
data, training budget, and validation identical. This is an encoder/decoder
adapter baseline, not a pure output-only corrector. Its gradients still traverse
the GCM, so it is not guaranteed to be substantially faster.

If adapters match the full model, report the result as neural forecast adaptation;
do not claim improved boundary-layer physics or learned dynamics. If the full
model improves both internal and decoded forecasts beyond the adapter baseline,
run `--ablation tendency-only` next to isolate learned atmospheric tendencies.
These runs are screening experiments; a broad hyperparameter search is outside
the deadline budget.

## Optional PBL diagnostic: no optimizer updates

The existing checkpoint can be screened on one GPU without restarting training:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_neural_pbl.py \
  --checkpoint outputs/neural_pbl/distributed/best_validation.pkl \
  --output outputs/neural_pbl/deadline_screen \
  --examples 4 --constant-screen
```

This performs 40 forward forecasts: the learned checkpoint plus a nine-point
constant grid, including the physical unit/unit baseline. Warmup is shared across
variants for each case. It reports global error, lowest-three-layer error,
per-layer RMSE, and per-case scores. The constants span approximately 0.5, 1, and
2 for momentum diffusivity and Prandtl-number multipliers.

If even these large changes have negligible effect, prioritize the broader hybrid
instead of spending hours adjusting the old PBL optimizer. A constant grid cannot
prove that a state-dependent closure has no possible benefit. Lowest-three-layer
metrics are not a diagnosed PBL-height mask. Selection on this small validation
sample is not an independent test result.

## Reserve the second day for evidence

Choose architecture/checkpoint using validation, then evaluate the frozen choice
on unseen test periods with a paired physical baseline. Record sample coverage,
excluded source windows, runtime, and per-field/layer errors. Retain time for
plots, reproducibility, and writing. Full test evaluation may be expensive:
estimate per-example cost before committing the remaining GPU allocation.

Do not present short-horizon prediction improvements as climate stability,
terraforming generalization, or improved physical mixing without direct evidence.
If neural gains remain weak, report them honestly rather than selecting test
cases or moving the evaluation criterion after seeing results.
