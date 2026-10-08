#!/usr/bin/env bash

set -euo pipefail

source /work/master-thesis/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cd "$DELTA_PROOF"
source "$DELTA_PROOF/.venv/bin/activate"

nvidia-smi -L
python --version

uv run --no-sync python -m alphaproof.training.sft \
    sft_codet5p_770m_full_b200_01 \
    alphaproof/yaml/UCloud/codet5p_770m_full_b200_sft.yaml
