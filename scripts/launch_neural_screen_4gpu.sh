#!/usr/bin/env bash
set -euo pipefail
# Four independent methods, one GPU each. Run from the repository root.
export PYTHONUNBUFFERED=1
exec "${NEURAL_HYBRID_PYTHON:-python}" -u scripts/run_neural_screen.py "$@"
