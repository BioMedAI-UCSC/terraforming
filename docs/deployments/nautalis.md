# Nautalis

The generic runtime in `apps/nautalis` runs experiment folders on the Nautilus
Kubernetes cluster. It contains no experiment suite or simulation scripts.
Create a folder from the template command, add your implementation, and commit
the folder to the branch or repository revision that the job will fetch.

## Create an experiment folder

From the repository root:

```bash
python3 apps/nautalis/init.py climate-study
```

This creates `experiments/climate-study/README.md` and `run.py`. The generated
`run.py` is an intentional failing stub until replaced with the experiment.
Use `--parent PATH` to place the folder elsewhere. The command refuses invalid
names and existing folders.

An experiment folder must provide `run.py` or `run.sh`. The runtime selects
`run.py` when both exist. It starts the entry point from the repository root
and forwards arguments after `--` unchanged. Keep dependencies and protocol
setup in the experiment folder; the image contains only the generic runtime.

## Submit a job

Commit the experiment folder, then submit it by branch, tag, or commit:

```bash
python3 apps/nautalis/jobs.py \
  --name climate-study \
  --ref YOUR_BRANCH \
  --experiments experiments/climate-study \
  --run-id run-1 \
  --submit -- --case baseline
```

Omit `--submit` to print the Kubernetes manifests without launching a job. Use
an immutable commit for repeatable source selection; the pod records and
archives the resolved source.

## Job inputs

| Argument | Required/default | Meaning |
| --- | --- | --- |
| `--name` | Required | Kubernetes Job name; also the default PVC name. |
| `--ref` | Required | Branch, tag or commit containing the experiment folder. |
| `--experiments` | Required | One or more folders relative to the fetched source root. |
| `--run-id` | Required | Caller-selected identity using letters, digits, dashes and underscores. |
| Arguments after `--` | Optional | Forwarded unchanged to each selected entry point. |
| `--repository` | This project's GitHub repository | Git repository to fetch. |
| `--gpu` | Off | Request one GPU and set `EXPERIMENT_GPU=1`. |
| `--cpu`, `--memory` | `8`, `64Gi` | CPU and memory requests/limits. |
| `--storage` | `500Gi` | Persistent volume capacity. |
| `--pvc` | Job name | Existing/shared claim name. |
| `--context`, `--namespace` | `nautilus`, `razvanlab` | Explicit Kubernetes target. |
| `--submit` | Off | Apply the rendered Job and PVC to Kubernetes. |

The defaults request one 500 GiB RWO volume, eight CPUs and 64 GiB RAM. Jobs run
as uid/gid 10001, with a read-only root filesystem, no service-account token, a
seven-day deadline and two retries. A file lock rejects concurrent writers to
the same work volume. Resource requests do not guarantee cluster capacity.

The published image pinned by the job tool is:

```text
samkash/terraforming-paper@sha256:a04422421dbe06a2073d55259ba9ae9d54e2f22488e4bc2552fdbb5bc3f24fd2
```

## Runtime contract

The entry point receives these environment variables:

| Variable | Meaning |
| --- | --- |
| `EXPERIMENT_SOURCE_DIR` | Resolved checkout or mounted source directory. |
| `EXPERIMENT_WORK_DIR` | Persistent scratch space for dependencies, data and explicitly managed state. |
| `EXPERIMENT_RUN_DIR` | Stable output root for the source/run identity; use it for validated checkpoint management. |
| `EXPERIMENT_OUTPUT_DIR` | Fresh output directory for this attempt's result files. |
| `EXPERIMENT_RUN_ID` | Requested run ID. |
| `EXPERIMENT_GPU` | `1` when a GPU was requested, otherwise `0`. |

Write results to `EXPERIMENT_OUTPUT_DIR`, not the source checkout. Retries
create new attempts and preserve earlier output. The launcher does not infer
checkpoint validity or scientific acceptance; each experiment owns those
checks. SIGTERM and SIGINT are forwarded to the experiment process group.

## Output and download

The persistent volume stores each run under:

```text
/work/outputs/EXPERIMENT/SOURCE_ID/RUN_ID/
  source.json
  source.tar
  attempts/TIMESTAMP/
    run.log
    status.json
    outputs/
```

Download a job's artifacts and pod records with:

```bash
python3 apps/nautalis/download.py --job JOB_NAME
```

Downloads are timestamped under `outputs/JOB_NAME/`; existing snapshots are
never overwritten. A completed job is read through a temporary read-only PVC
reader, which is removed afterward. Live downloads are partial snapshots, and
file hashes do not establish scientific correctness.

Established research results and their interpretation are documented separately
in the [curated result archive](../../iclr-results/README.md). It records
selected outcomes and limitations, rather than bundling the simulation suite
into this generic deployment runtime.
