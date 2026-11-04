#!/usr/bin/env bash
# Run once on the NVIDIA GPU server, from this checkout. No model run here.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
nvidia-smi
uv sync --python 3.12 --frozen --dev --all-packages --extra gcm3d
aerosol_jax_version=$(.venv/bin/python -c 'import jax; print(jax.__version__)')
# Preserve the lockfile's JAX version; add its CUDA 12 runtime/plugin wheels.
# Use .venv/bin/python afterward: an ordinary uv run/sync can restore CPU JAX.
uv pip install --python .venv/bin/python "jax[cuda12]==${aerosol_jax_version}"
JAX_PLATFORMS=cuda .venv/bin/python -c 'import jax; print("JAX",jax.__version__); print(jax.devices()); assert any(d.platform=="gpu" for d in jax.devices())'
echo 'GPU environment ready. Use .venv/bin/python or scripts/run_aerosol_gpu.sh; avoid re-syncing before a run.'
