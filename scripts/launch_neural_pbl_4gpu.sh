#!/usr/bin/env bash
set -euo pipefail
# One process, four devices, one synchronized optimizer. Do not use torchrun.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec rtk proxy "${NEURAL_PBL_PYTHON:-.venv/bin/python}" scripts/train_neural_pbl.py \
  --revision 65a0bebd804b9c240752277e83f5737d58c6ee9c \
  --output outputs/neural_pbl/distributed --devices 4 "$@"
