# ICLR experiments on Nautilus

The generic image and Kubernetes tools live in `apps/nautalis`. This folder
owns the ICLR protocols, data staging, pinned scientific revisions, runtime
dependencies, retries and downloads. `experiment_tasks.py` lists every command;
`run.py` runs `all` by default, or `stage`, `list`, or one named task.

`run.py:REVISIONS` retains the exact main baseline and PR #115–117 revisions.
Changing the launcher branch selects the experiment definition; it does not
silently replace these frozen scientific revisions. Source merges still use
fixed dates and preserve the source archive and environment inventory.

## Run the suite

From the repository root:

```sh
rtk proxy python3 iclr_experiments/run.py list
rtk proxy python3 apps/nautalis/jobs.py --name iclr-all \
  --ref feat/nautilus-paper-experiments --experiments iclr_experiments \
  --run-id paper-v2 --gpu --submit -- all
```

This creates a new PVC named `iclr-all` and an A100 job. It does not replace
legacy `paper-all` or its PVC. CPU input staging runs first inside this job
when needed, so the GPU is reserved during data preparation. To avoid that,
submit `stage` without `--gpu` first, wait for it to finish, then submit `all`
with `--gpu`, the same explicit `--pvc`, immutable `--ref` commit and `--run-id`.
Do not run concurrent writers against one PVC.

The suite owns installation (`uv sync --frozen` with gcm3d/arco and pinned
CUDA JAX). The image contains no JAX or experiment code. GPU fallback to CPU
is rejected. Staging retains completed steps and renames interrupted exclusive
outputs instead of overwriting them. Successful tasks are skipped on retry;
failed attempts remain, climate resumes its latest checkpoint, and a failed
experiment makes the suite fail after other tasks are attempted.

## Outputs and default download

All generated results, prepared input manifests, source/environment records
and logs are under `/work/outputs/iclr_experiments/LAUNCHER_COMMIT/RUN_ID/`:

- `source.json`, `source.tar`: experiment-definition checkout provenance.
- `attempts/`: every pod invocation and its streamed log/status.
- `prepared/SCIENCE_ID/`: staged generated inputs/windows and their manifests.
- `tasks/SCIENCE_ID/`: scientific source archive, environment and per-task attempts.

Raw downloaded datasets and the virtualenv remain cached in the persistent
work area. They are not part of a default result-only download.

```sh
rtk proxy python3 iclr_experiments/download.py
# Or another job / a specific snapshot location:
rtk proxy python3 iclr_experiments/download.py --job iclr-all \
  --output outputs/iclr_experiments/paper-v2-snapshot
```

The default job is `iclr-all`. The downloader saves pod/job JSON, pod logs,
results and local SHA256 hashes, and cleans up its temporary read-only transfer
pod. Each default download creates a timestamped directory; nonempty explicit
destinations are rejected. Downloads from running jobs are partial snapshots.

Legacy PR #118 results are still available separately:

```sh
rtk proxy python3 iclr_experiments/download.py --job paper-all \
  --path /work/runs --output outputs/iclr_experiments/legacy-paper-v1
```

The legacy PVC and current cluster jobs are unchanged by this refactor. Keep
all PVCs until source, inputs and scientific evidence have been archived.

## Protocols and scientific limits

| Task | Paper evidence |
| --- | --- |
| radiation-0, radiation-repeat, radiation-1, radiation-2 | Fixed 100/20/20-update protocols, independent seed-0 repeat and two more seeds |
| column-recovery, control | 120-iteration scalar recovery; corrected 40-iteration neural control |
| preflight, gradients | Native-window validation; AD/FD horizons 1–148 and three perturbation sizes |
| scalar-sensitivity | Multi-diagnostic derivatives at 74/296/1480 steps |
| calibration-0/1/2 | Each: three bounded starts, 500 updates / 1000 evaluations |
| neural-0/1/2 | 50-update native-window neural pilots |
| short-ablations | No PBL/convection/regolith and half timestep |
| synthetic, timing | Three-control recovery and synchronized benchmark |
| annual | All cases in the existing Mars-year **300/150 s** ablation protocol |
| climate | Untuned T21L12 **2006-sol** seasonal spinup; checkpoint every five sols |
| seasonal | Global and spatial year-2/year-3 convergence; four seasonal exports; Ames, held-out MY34/35 ARCO, and online MCD comparisons |

CPU staging downloads MOLA, TES, pinned ARCO-MACDA native training/validation/test
windows and seasonal references, and the Ames raw NetCDF. It creates fresh
physical restarts and the frozen input manifest with the current PR physics.
Completed staging steps are retained; interrupted exclusive-output directories
are renamed `.partial-*` instead of overwritten. Runtime requires internet
access to GitHub, PyPI, Hugging Face, NASA/PDS, and the MCD service.

All GPU tasks execute sequentially. A failing experiment is recorded and the
runner attempts the other tasks; seasonal evaluation requires a successful
climate simulation. Any failure makes the overall Job fail. A zero process exit
is **not** scientific acceptance: review each experiment's acceptance/convergence,
stationarity, uncertainty and failed-column reports before updating the paper.
Seasonal sampling/height/dust limitations remain in the benchmark reports.

This runs the implemented reproduction protocols, not missing experiments or
physics: atmospheric CO₂ condensation, true common-state 450/300/finer timestep
certification, raw energy/momentum closure and default T170 certification remain
open. The 2006-sol run is a convergence pilot, not an automatic ten-year or
equilibrated-climate claim. Missing historical raw headline fields cannot be
recreated by copying the archived small evidence. Extend climate runs only after
reviewing convergence and preserving an identical physical configuration.

NRP references: [GPU resources](https://nrp.ai/documentation/userdocs/running/gpu-pods/),
[Jobs](https://nrp.ai/documentation/userdocs/running/jobs/),
[storage](https://nrp.ai/documentation/userdocs/storage/intro/).
