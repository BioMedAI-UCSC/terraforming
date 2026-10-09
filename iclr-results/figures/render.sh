#!/usr/bin/env bash
# Regenerate every paper figure (PNG + PDF) into iclr-results/figures/out/.
# Run from the repository root:  bash iclr-results/figures/render.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

echo "Using interpreter: $PY"
"$PY" iclr-results/figures/make_figures.py all "$@"
