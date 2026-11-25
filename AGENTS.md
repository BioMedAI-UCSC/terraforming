# Codex working rules

Apply Ponytail full mode. Inspect affected code, callers, tests and configuration
before editing. Make the smallest complete change. Preserve validation, error
handling and data safety. Add a focused check for non-trivial logic. Report work
that was not checked and any material risk.

## Clean Code before pushing

- Push only changes that pass a Clean Code review of the full outgoing diff.
  Use clear domain names, focused functions with one responsibility, simple
  control flow and existing conventions. Remove dead code, avoidable duplication
  and speculative abstractions. Fix review findings before pushing.
- Pass the applicable tests, lint and formatting checks before pushing.
- Commit reusable code, tests, fixtures and documentation. Keep one-off experiment
  manifests, scripts, generated results and logs in CI artifacts or PR summaries.

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

## Agent skills

### Issue tracker

Track issues and specs in GitHub Issues for `BioMedAI-UCSC/terraforming`.
Read `docs/agents/issue-tracker.md` before issue operations.

### Triage labels

Use the five standard triage labels. Read `docs/agents/triage-labels.md`
before assigning triage states.

### Domain docs

Use one root glossary and `docs/adr/`. Read `docs/agents/domain.md`
before domain exploration or recording design decisions.
