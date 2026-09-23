#!/usr/bin/env bash
set -euo pipefail
# Run from the repo in the activated mamba environment. One optimizer, four GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
exec "${NEURAL_HYBRID_PYTHON:-python}" -u scripts/train_neural_hybrid.py \
  --revision 65a0bebd804b9c240752277e83f5737d58c6ee9c \
  --mola "${MOLA_PATH:-outputs/nautilus-calibration-inputs/mola.img}" \
  --output outputs/neural_hybrid/distributed --devices 4 "$@"
