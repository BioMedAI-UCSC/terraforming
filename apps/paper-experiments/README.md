# Paper experiments on Nautilus

Namespace **razvanlab**, context **nautilus**, Docker Hub **samkash**.
The existing short-calibration application is unchanged. This application runs
the current paper protocols with the combined, pinned code from PRs **115–117**.
You do not need to merge the PRs before running. `run.py:REVISIONS` records the
exact main baseline and three PR commits; runtime merges use fixed commit dates.

## Run everything

From this checkout's repository root, with Python 3, `rtk`, and authenticated
`kubectl` on your laptop:

```sh
rtk proxy sh apps/paper-experiments/launch.sh
```

Keep that command running until it submits `paper-all`. It first creates a
**500 GiB persistent volume** and a CPU staging Job. After staging succeeds it
submits the A100 Job; the laptop can then disconnect. No local simulation or
local data copy is used. Both Jobs have a seven-day deadline and two pod retries.
The launch script explicitly targets Nautilus and never changes your context.
If staging fails, stop the wait with Ctrl-C and inspect the logs below.

```sh
rtk kubectl --context nautilus -n razvanlab get jobs,pods,pvc
rtk kubectl --context nautilus -n razvanlab logs -f job/paper-stage
rtk kubectl --context nautilus -n razvanlab logs -f job/paper-all
```

Only one Job may use this volume at a time. The volume is RWO; a runtime file
lock also rejects concurrent writers. Delete the results-transfer pod before
starting another computation. Compute requests/limits are 8 CPU and 64 GiB RAM,
plus one `nvidia.com/a100` for the GPU Job. It uses normal priority because this
namespace has A100 quota; quota does not guarantee immediate GPU availability.

## What runs

`rtk proxy python3 apps/paper-experiments/run.py list` prints the exact commands.

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

## Small image, runtime installation and compilation

Published Linux/amd64 image:

```
samkash/terraforming-paper@sha256:47ef0396b05394d0d19395c99ac54f058356c486b8def7213669641cfd17a7f3
```

Built from Python 3.12 slim with Git, CA certificates and pinned `uv` 0.10.9.
Docker reports 97,120,652 bytes for this image. The build context contains only
the bootstrap and runner. No repository, datasets, Torch, JAX or CUDA libraries
are baked in. Runtime installs the repository's frozen lockfile with `gcm3d`
and `arco`, then the pinned `jax[cuda12]==0.11.0` on the GPU pod. CUDA userspace
libraries come from wheels, the NVIDIA driver comes from the node, and JAX JIT
compilation happens on the allocated hardware. No system compiler is required
for these wheel-based protocols. Dependencies occupy many GiB at runtime.

Source, virtualenv, download cache, JAX compilation cache, data and results live
on the PVC. GPU fallback to CPU is rejected. Each pod startup syncs the runtime
environment and installs CUDA afterward; commands then use `.venv/bin/python`
directly so another sync cannot remove CUDA wheels mid-experiment.

To rebuild after changing the runner, use a new image tag and pass its pushed
digest to `jobs.py --image` (the default digest always refers to this release):

```sh
rtk proxy docker build --platform linux/amd64 -t samkash/terraforming-paper:YOUR_TAG apps/paper-experiments
rtk proxy docker push samkash/terraforming-paper:YOUR_TAG
```

## Run one task or retry

Render, review, validate and submit a selected task, after staging is complete:

```sh
rtk proxy python3 apps/paper-experiments/jobs.py gradients --run-id paper-v1 > /tmp/paper-job.json
rtk kubectl --context nautilus -n razvanlab apply --dry-run=server -f /tmp/paper-job.json
rtk kubectl --context nautilus -n razvanlab apply -f /tmp/paper-job.json
```

Do not run this while `paper-all` is active. Finished Jobs keep their pods for
inspection. To repeat a completed/failed Job, delete **only that finished Job**
(e.g. `rtk kubectl --context nautilus -n razvanlab delete job paper-all`) and
submit it again. Keep the PVC. Under the same run ID, successful tasks are
skipped, failed tasks get fresh timestamped attempts, and climate resumes the
last saved checkpoint. A new `--run-id` repeats every selected experiment with
fresh outputs. It does not redownload the same source-specific inputs.
The long annual ablation script has no checkpoint resume; preemption restarts
that task. Failed scientific acceptance should be reviewed before retrying.

## Collect results and update the paper

Each attempt has a persistent `run.log` and `results/`. Task status JSON files,
the source revision list, complete merged `source.tar`, and installed package /
JAX-device inventory are saved under `/work/runs/SOURCE_ID/RUN_ID/`.
The checkout plus immutable input contracts/manifests remain under
`/work/source/SOURCE_ID/`. Pod logs print the full result paths. For live logs,
use `kubectl exec` into the running pod and `tail` the printed `run.log` path.

After computation stops and its pods are gone, mount the PVC for download:

```sh
rtk kubectl --context nautilus -n razvanlab apply -f apps/paper-experiments/results-pod.yaml
rtk kubectl --context nautilus -n razvanlab wait --for=condition=Ready pod/paper-results --timeout=5m
rtk kubectl --context nautilus -n razvanlab cp paper-results:/work/runs ./nautilus-paper-results
```

Archive inputs and source too before deleting the PVC. Review new reports, then
follow `docs/paper-reproduction.md` from PR #117 to replace the relevant frozen
evidence and update `claims.json`. `regenerate.py` verifies/rebuilds frozen
evidence; it does **not** automatically import new simulation directories.
Paper TeX/PDF compilation is a separate manuscript step and is not included in
the small simulation image.

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
