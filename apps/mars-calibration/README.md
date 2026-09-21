# Short-horizon Mars calibration on Nautilus

For the separate trajectory-recovery, controlled-ablation, and timing workflow,
see [the paper experiment guide](../../docs/ideas/paper-experiments.md).
`paper-experiment.json` and `paper-smoke.json` do not change the frozen Phase-1
`experiment.json` protocol described below.

This standalone application installs the repository framework as
`terraforming[gcm3d]`. It runs a real coupled **0.25-sol / 74-step** rollout at
**T21/L12, dt=300 s, float64**, retaining diurnal/orbital forcing, correlated-k
radiation, seasonal dust, TES surfaces, MOLA, regolith, exchange, PBL mixing,
convection, spectral diffusion, and the conservative CO2 cycle.

The fixed experiment fits CO2 longwave opacity, dust longwave opacity, and surface
exchange from `(0.25, 0.25, 1)`. Both optimizers get exactly 20 oracle calls.
An early exit restarts the optimizer from its best evaluated point; restart
counts are reported. These are equal call budgets, not equal FLOPs or elapsed
time. Two finite-difference step sizes × three parameters × two signs require
12 additional forward calls, outside optimization. Every gradient component
must agree within 1e-4 relative error at both step sizes. Success additionally
requires lower loss than initialization and loss no worse than Powell.

The single MY32 Ls45 target is index 5 of a daily-mean reanalysis file. The model
endpoint is instantaneous, and this temporal mismatch is disclosed. This is a
short-horizon calibration demonstration, not equilibrium tuning, held-out
validation, or a four-season calibration result. Actual integrated duration
(74 × 300 seconds) is reported separately from requested 0.25 sol.

## Prepare and build

From the repository root, freeze the existing inputs (already prepared for this
checkpoint at `outputs/nautilus-calibration-inputs`):

```sh
rtk proxy .venv/bin/python apps/mars-calibration/prepare_inputs.py --output outputs/nautilus-calibration-inputs
rtk proxy env PYTHONPATH=apps/mars-calibration .venv/bin/python -m mars_calibration.launch --config apps/mars-calibration/experiment.json --inputs outputs/nautilus-calibration-inputs --output /tmp/unused --validate-only
```

Preparation refuses to overwrite a bundle. Five files plus their SHA-256
manifest are sufficient; the full archives are not needed. Default source
paths are in `prepare_inputs.py`. Build from the current working tree so the
framework includes the current physics fixes, including uncommitted changes:

```sh
rtk proxy docker buildx build --platform linux/amd64 -f apps/mars-calibration/Dockerfile -t YOUR_REGISTRY/mars-calibration:short-horizon --push .
```

Replace `YOUR_REGISTRY` with a registry you can push to and Nautilus can pull
from. Use the resulting immutable image digest in `job.yaml`. The image installs
CUDA 12 JAX 0.11.0 matching the local JAX version; the node must provide a
compatible NVIDIA driver. The host CUDA toolkit is not used. `pip freeze` is
saved in the image and copied into results. A build-specific Docker ignore file
excludes datasets, outputs, and caches from the context.

## Transfer and submit

Use your existing Nautilus namespace and credentials; replace `YOUR_NAMESPACE`
below. The checked-in Job follows the current NRP/Nautilus specifications for
an opportunistic A100 workload:

| Resource | Request | Limit |
|---|---:|---:|
| NVIDIA A100 | 1 | 1 |
| CPU | 4 cores | 8 cores |
| Host memory | 16 GiB | 32 GiB |
| Ephemeral storage | 2 GiB | 8 GiB |
| `/dev/shm` | 8 GiB memory-backed volume | 8 GiB |
| Persistent storage | 10 GiB RBD PVC | 10 GiB |

NRP advertises A100s as `nvidia.com/a100`; `nvidia.com/gpu` is the generic GPU
resource and must not be used when requesting this special GPU type. A100 quota
is zero by default. The manifest therefore uses `priorityClassName:
opportunistic`, which bypasses that quota but permits preemption at any time. It
retries twice, and each attempt writes to `/results/calibration-<pod-name>` so a
retry cannot overwrite a partial attempt. If your namespace receives approved
A100 quota, remove `priorityClassName` to run at normal priority. Do not replace
it with another priority class unless NRP documents that class as user-allowed.
Exit code 10 means the experiment completed but failed its scientific acceptance
criteria; `podFailurePolicy` marks that result final instead of repeating it.

The Job is restricted to US West because the explicit `rook-ceph-block` PVC is
in that region. RBD is the NRP-recommended storage for a volume mounted by one
pod at a time. The Job also requires an NVIDIA driver major version above 524,
which is suitable for the CUDA 12 JAX image. Before submission, inspect current
labels and availability:

```sh
rtk proxy kubectl get nodes \
  -L topology.kubernetes.io/region,nvidia.com/gpu.product,nvidia.com/cuda.driver.major \
  -l nvidia.com/gpu.product
rtk proxy kubectl -n YOUR_NAMESPACE get resourcequota
```

The container runs as UID/GID 10001 with no service-account token, no added
Linux capabilities, a read-only root filesystem, and writable `/tmp`, `/dev/shm`
and result volumes. The image is still a placeholder: replace it with a pushed
image digest or immutable tag, and configure `imagePullSecrets` if the registry
is private.

```sh
rtk proxy kubectl -n YOUR_NAMESPACE apply -f apps/mars-calibration/pvc.yaml
rtk proxy kubectl -n YOUR_NAMESPACE apply -f apps/mars-calibration/data-pod.yaml
rtk proxy kubectl -n YOUR_NAMESPACE wait --for=condition=Ready pod/mars-calibration-data --timeout=120s
rtk proxy kubectl -n YOUR_NAMESPACE cp outputs/nautilus-calibration-inputs/. mars-calibration-data:/data/inputs/
rtk proxy kubectl -n YOUR_NAMESPACE delete pod mars-calibration-data --wait=true
rtk proxy kubectl -n YOUR_NAMESPACE apply -f apps/mars-calibration/job.yaml
rtk proxy kubectl -n YOUR_NAMESPACE logs -f job/mars-short-calibration
```

The transfer pod is removed before the GPU job to avoid concurrent mounts of a
ReadWriteOnce volume. It has a one-hour deadline, so recreate it if a transfer
takes longer. The scientific Job retries at most twice after a nonzero exit or
preemption. Compilation can take time before the first evaluation appears. CPU
fallback is forbidden. The Job has a 24-hour total deadline. Results go to a
pod-specific directory under `/results`, including partial attempts.

After the job completes or fails, retain the logs, delete the finished job to
release its mount, then recreate the transfer pod and retrieve artifacts:

```sh
rtk proxy kubectl -n YOUR_NAMESPACE logs job/mars-short-calibration --all-containers > outputs/nautilus-calibration.log
rtk proxy kubectl -n YOUR_NAMESPACE delete job mars-short-calibration --wait=true
rtk proxy kubectl -n YOUR_NAMESPACE apply -f apps/mars-calibration/data-pod.yaml
rtk proxy kubectl -n YOUR_NAMESPACE wait --for=condition=Ready pod/mars-calibration-data --timeout=120s
rtk proxy kubectl -n YOUR_NAMESPACE cp mars-calibration-data:/data/results outputs/nautilus-results
rtk proxy kubectl -n YOUR_NAMESPACE delete pod mars-calibration-data --wait=true
```

Inspect `report.json` for scientific acceptance, exact counts, finite-difference
errors, best parameters, losses, and per-call timings (first calls include JIT
compilation). `report.jsonl` preserves completed evaluations during execution.
`input-manifest.json`, `experiment.json`, and `environment.txt` preserve input
and runtime provenance. Keep the image digest with the downloaded artifacts.
No cluster workload has been submitted or GPU calibration result claimed here.

The resource syntax and scheduling choices were checked against the official
[NRP GPU Pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/),
[Batch Jobs](https://nrp.ai/documentation/userdocs/running/jobs/),
[Priority Classes](https://nrp.ai/documentation/userdocs/running/priority-classes/),
and [Storage](https://nrp.ai/documentation/userdocs/storage/intro/) documentation.

## Local checks

```sh
rtk proxy .venv/bin/python -m pytest apps/mars-calibration/tests -q
```

These tests use a cheap analytic oracle for optimizer accounting, not a surrogate
for the production experiment. The legacy `scripts/calibrate_mars_physics.py`
entry point permits explicitly smaller CPU smoke runs; the GPU application
locks the canonical experiment to 74 steps and 20/20 calls.
