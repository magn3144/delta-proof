#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cd "$DELTA_PROOF"
source scripts/batch/UCloud/common_b200.sh

uv run --no-project --active "$UV_PROJECT_ENVIRONMENT/bin/python" -m alphaproof.training.sft \
    sft_codet5p_770m_full_b200_01 \
    alphaproof/yaml/UCloud/codet5p_770m_full_b200_sft.yaml
