# Generic experiment image and Nautilus jobs

The `nautalis` runtime deploys application and experiment folders independently
of their protocols or source revisions. The small Linux/amd64 image provides Python 3.12, Git, CA
certificates, tar and pinned uv 0.10.9; experiment folders own their setup and
execution.

Published image (the default pinned by `jobs.py`):
`samkash/terraforming-paper@sha256:a04422421dbe06a2073d55259ba9ae9d54e2f22488e4bc2552fdbb5bc3f24fd2`.

## Run folders from a GitHub branch

Create a blank experiment folder in your checkout, implement its `run.py`, and
commit it before submitting:

```sh
python3 apps/nautalis/init.py climate-study
```

This creates `experiments/climate-study/README.md` and `run.py`. The starter
entry point fails until you replace it with the experiment implementation.

```sh
python3 apps/nautalis/jobs.py --name climate-study \
  --ref YOUR_BRANCH --experiments experiments/climate-study \
  --run-id run-1 --gpu --submit -- arg1
```

`--ref` accepts a branch, tag or immutable commit. The pod resolves and archives
the actual commit before execution. `--repository` can select another Git
repository. Use `--experiments folder_a folder_b` for sequential execution;
arguments after `--` are passed unchanged to each selected entry point.
Without `--submit`, `jobs.py` only renders JSON for review/server dry-run.

Context `nautilus` and namespace `razvanlab` are explicit defaults; neither tool
changes your current context. Override them with `--context`/`--namespace`.
The default job requests 8 CPUs, 64 GiB RAM and a 500 GiB RWO PVC named after
the job. `--gpu` additionally requests one A100; CPU, memory, storage, PVC name
and GPU resource type can be set explicitly. The job has a seven-day deadline,
two retries, non-root execution, no service-account token and a read-only root
filesystem. Concurrent writers to the same PVC are rejected by a file lock.

## Run a mounted experiment folder

```sh
docker run --rm --platform linux/amd64 \
  -v /ABSOLUTE/SOURCE:/source:ro -v /WRITABLE/WORK:/work \
  samkash/terraforming-paper:nautalis-v1 \
  --source /source --experiments my_experiment --run-id local-1 -- arg1
```

Use `--experiments .` when the mount itself is one experiment folder. The work
mount must be writable by uid/gid 10001 and outside the source mount. Mounted
source is content-hashed and archived, excluding `.git`, `.venv`, `outputs` and
Python caches. Symlinks are archived as links rather than followed.

## Experiment folder contract

A folder supplies `run.py` (Python entry point) or `run.sh` (POSIX shell entry
point). If both exist, `run.py` is selected. Each entry point installs its own
locked dependencies if needed and returns nonzero on failure. It runs with the
repository root as its working directory and receives:

- `EXPERIMENT_SOURCE_DIR`: resolved checkout or mounted source.
- `EXPERIMENT_WORK_DIR`: persistent scratch/environment/data area for this run.
- `EXPERIMENT_RUN_DIR`: stable run root for explicitly managed checkpoints/state.
- `EXPERIMENT_OUTPUT_DIR`: fresh output directory for this invocation.
- `EXPERIMENT_RUN_ID`: caller-selected run identity.
- `EXPERIMENT_GPU`: `1` when GPU execution was requested, otherwise `0`.

Do not write generated files into the checkout. Generic retries always create a
new attempt and keep earlier outputs. An experiment that resumes checkpoints
must validate their identity and use `EXPERIMENT_RUN_DIR`/`EXPERIMENT_WORK_DIR`;
the launcher does not guess how to resume arbitrary science code. SIGTERM and
SIGINT are forwarded to the experiment process group for graceful checkpointing.

Output layout on the PVC:

```text
/work/outputs/EXPERIMENT/SOURCE_ID/RUN_ID/
  source.json
  source.tar
  attempts/TIMESTAMP/run.log
  attempts/TIMESTAMP/status.json
  attempts/TIMESTAMP/outputs/
```

## Download outputs and pod records

```sh
python3 apps/nautalis/download.py --job JOB_NAME
```

This writes a timestamped snapshot under local `outputs/JOB_NAME/`, including
pod logs, job/pod specifications, run artifacts and local SHA256 hashes. It uses
a running job pod when possible, or a temporary read-only PVC reader for a
finished job, then removes that reader. Existing snapshots are never overwritten.
The hashes describe downloaded files; they do not certify an atomic live
snapshot or replace scientific validation. Do not concurrently use a separate
writer job on the same RWO claim during a transfer.

## Build and test

```sh
python3 apps/nautalis/test_runtime.py
docker build --platform linux/amd64 \
  -t samkash/terraforming-paper:nautalis-v1 apps/nautalis
docker push samkash/terraforming-paper:nautalis-v1
```

Only bootstrap/runtime code is baked into the image. New experiment folders or
branches require no image rebuild; changes to the generic runtime do. Jobs pin
the published image digest once built. Existing jobs and their persistent
volumes remain independent.
