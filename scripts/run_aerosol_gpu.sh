#!/usr/bin/env bash
# Usage: bash scripts/run_aerosol_gpu.sh OUTPUT_DIR [--case NAME ...]
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run_aerosol_gpu.sh OUTPUT_DIR [--case NAME ...]' >&2
  exit 2
fi
aerosol_output=$1
shift
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set +e
.venv/bin/python scripts/aerosol_paper_experiments.py run --output "$aerosol_output" "$@"
aerosol_run_status=$?
# Also make partial plots after a graceful stop or a validity-gate failure.
.venv/bin/python scripts/aerosol_paper_experiments.py plot --output "$aerosol_output"
aerosol_plot_status=$?
set -e
if [[ $aerosol_run_status -ne 0 ]]; then exit "$aerosol_run_status"; fi
exit "$aerosol_plot_status"
