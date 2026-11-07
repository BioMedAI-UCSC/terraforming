# Nautalis

The `apps/nautalis` tools run experiment folders on the **Nautilus** Kubernetes
cluster. The folder and deployment option use the requested name **Nautalis**;
the Kubernetes context is `nautilus`.

The image supplies Python, Git, tar, CA certificates and uv. It fetches the
selected source and runs an experiment entry point. Experiment folders install
their own dependencies and own their scientific protocols. New experiment
folders do not require rebuilding the image.

## Submit a job

Run from the repository root with Python 3, RTK and authenticated `kubectl`:

```bash
rtk proxy python3 apps/nautalis/jobs.py \
  --name iclr-all \
  --ref feat/nautilus-paper-experiments \
  --experiments iclr_experiments \
  --run-id paper-v2 \
  --gpu --submit -- all
```

Here, `all` is passed to the ICLR experiment entry point. Omit `--submit` to
render the Kubernetes JSON without launching anything. Use an immutable commit
instead of a branch for a repeatable source selection: a branch is resolved
again when a new pod starts, and a changed commit gets a different output root.

## Job input contract

| Argument | Required/default | Meaning |
| --- | --- | --- |
| `--name` | Required | Kubernetes Job name; also the default PVC name. |
| `--ref` | Required | Branch, tag or commit containing the experiment folders. |
| `--experiments` | Required | One or more folders relative to the source root, run sequentially. `--experiment` is an alias. |
| `--run-id` | Required | Caller-selected run identity, using letters, digits, dashes and underscores. |
| Arguments after `--` | Optional | Forwarded unchanged to every selected experiment entry point. |
| `--repository` | This project's GitHub repository | Git repository to fetch. |
| `--gpu` | Off | Request one GPU and set `EXPERIMENT_GPU=1`. |
| `--gpu-resource` | `nvidia.com/a100` | Kubernetes GPU resource type when `--gpu` is set. |
| `--cpu`, `--memory` | `8`, `64Gi` | CPU and host memory requests/limits. |
| `--storage` | `500Gi` | Requested persistent-volume capacity. |
| `--pvc` | Job name | Use an explicitly named claim, for example when staging and computation share data. |
| `--context`, `--namespace` | `nautilus`, `razvanlab` | Explicit Kubernetes target; the tool does not change the current context. |
| `--image` | Published digest below | Container image to run. |
| `--submit` | Off | Apply the rendered Job and PVC to Kubernetes. |

The default placement is `us-west` using `rook-ceph-block` storage. Jobs run
as uid/gid 10001, with a read-only root filesystem, no service-account token,
a seven-day deadline and two retries. A file lock rejects concurrent writers
against the same work volume. Resource requests do not guarantee immediate
cluster capacity; check scheduling and logs:

```bash
rtk kubectl --context nautilus -n razvanlab get job iclr-all
rtk kubectl --context nautilus -n razvanlab logs -f job/iclr-all
```

The published image pinned by the job tool is:

```text
samkash/terraforming-paper@sha256:a04422421dbe06a2073d55259ba9ae9d54e2f22488e4bc2552fdbb5bc3f24fd2
```

Its tag is `samkash/terraforming-paper:nautalis-v1`. CUDA libraries and experiment
packages are installed by the experiment, rather than baked into this image.
GPU experiments must verify that their runtime can see the requested device.

## Experiment-folder contract

Each selected folder supplies **`run.py` or `run.sh`**. If both exist, `run.py`
is chosen. The launcher invokes Python or POSIX shell respectively, with the
repository root as the working directory and the forwarded arguments unchanged.
Folders must resolve inside the supplied source directory.

| Environment variable | Meaning |
| --- | --- |
| `EXPERIMENT_SOURCE_DIR` | Resolved checkout or mounted source directory. |
| `EXPERIMENT_WORK_DIR` | Persistent scratch space for dependencies, data and explicitly managed state. |
| `EXPERIMENT_RUN_DIR` | Stable output root for the source/run identity; use it for validated checkpoint management. |
| `EXPERIMENT_OUTPUT_DIR` | Fresh output directory for this invocation. Write ordinary result files here. |
| `EXPERIMENT_RUN_ID` | Requested run ID. |
| `EXPERIMENT_GPU` | `1` when GPU execution was requested, otherwise `0`. |

The entry point installs its own locked dependencies, writes generated files
outside the source checkout, and returns nonzero on failure. SIGTERM and SIGINT
are forwarded to its process group so it can save a checkpoint before shutdown.
The launcher preserves attempts; it does not guess how arbitrary experiments
should resume or whether their scientific results are acceptable.

The ICLR suite in `iclr_experiments` provides `run.py`, `experiment_tasks.py`
and `download.py`. Its default task is `all`; it also supports `stage`, `list`
and individual named tasks. It owns its frozen scientific revision list,
staging and dependency installation. Selecting a launcher branch does not
silently replace that frozen scientific source. It retains successful tasks,
preserves failed attempts and resumes climate checkpoints with matching identity.
If inputs are missing, `all` stages them inside the GPU job. To stage on CPU
first, finish a separate `stage` job before submitting `all` with the same
explicit PVC, immutable source commit and run ID.

## Run mounted source with the image

The image also accepts a mounted source directory instead of `--ref`:

```bash
rtk proxy docker run --rm --platform linux/amd64 \
  -v /ABSOLUTE/SOURCE:/source:ro \
  -v /WRITABLE/WORK:/work \
  samkash/terraforming-paper:nautalis-v1 \
  --source /source --experiments my_experiment --run-id local-1 -- arg1
```

Use `--experiments .` when the mount itself is an experiment folder. The work
mount must be writable by uid/gid 10001 and outside the source directory.
Mounted source is hashed and archived, excluding `.git`, `.venv`, `outputs`
and Python caches; Git source is identified by its resolved commit.

## Output contract

Each experiment and source/run identity has a separate persistent output root:

```text
/work/outputs/EXPERIMENT/SOURCE_ID/RUN_ID/
  source.json
  source.tar
  attempts/TIMESTAMP/
    run.log
    status.json
    outputs/
```

`source.json` records the requested source and resolved revision. `source.tar`
archives the source. Every attempt has its streamed stdout/stderr, command,
return code and fresh result directory. A retry creates another attempt;
earlier attempt outputs remain available.

The ICLR suite additionally manages `prepared/SCIENCE_ID/` for generated input
bundles and `tasks/SCIENCE_ID/` for its scientific source/environment records
and per-task results. Downloaded raw datasets and virtualenvs remain in the
persistent work/cache area and are not included in a default result-only download.

## Download results and pod logs

For the default ICLR job:

```bash
rtk proxy python3 iclr_experiments/download.py
```

Or select a job and a new local snapshot directory:

```bash
rtk proxy python3 iclr_experiments/download.py \
  --job iclr-all --output outputs/iclr_experiments/paper-v2-snapshot
```

For any experiment job:

```bash
rtk proxy python3 apps/nautalis/download.py --job JOB_NAME
```

The ICLR helper defaults to job `iclr-all` and the remote ICLR output tree;
the generic helper downloads `/work/outputs`. Defaults create timestamped local
snapshots under `outputs/iclr_experiments/JOB/` or `outputs/JOB/` respectively.
Each snapshot contains:

```text
job.json
pods/POD.json
pods/POD.log
runs/
sha256.json
```

The downloader uses a running compute pod when possible. For a finished job,
it creates a temporary read-only PVC reader and removes it afterward. Nonempty
explicit destinations are rejected to preserve earlier downloads. Live downloads
are partial snapshots; local hashes describe the files downloaded and do not
certify an atomic remote snapshot or scientific correctness. Avoid a concurrent
writer on the same RWO claim during a transfer, and retain production PVCs until
the required evidence and inputs have been archived.
