# Codex working rules

Apply Ponytail full mode. Inspect affected code, callers, tests and configuration
before editing. Make the smallest complete change. Preserve validation, error
handling and data safety. Add a focused check for non-trivial logic. Report work
that was not checked and any material risk.

## Nautilus deployments

- Run independent experiments in parallel. Use concurrent experiments or an
  experiment matrix for every deployment. Size concurrency from measured GPU
  memory use and monitor GPU utilization. Increase useful parallel work when
  capacity remains, so experiments finish faster and keep GPUs actively used.
- Download sources, install dependencies and prepare inputs on CPU resources
  before starting GPU computation. Use CPU-only resources for Fortran or other
  workloads that do not use a GPU.
- Do not set Job or Pod deadlines, use timeout wrappers, or add automatic timed
  deletion. Do not stop running experiments because a time estimate or compute
  budget was reached. Let them finish unless the user asks to stop them.
