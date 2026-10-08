#!/usr/bin/env bash

set -euo pipefail

source "/work/MagnusElgaardBennett#3797/master-thesis/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cd "$DELTA_PROOF"
source "$DELTA_PROOF/.venv/bin/activate"

nvidia-smi -L
python --version

uv run --no-sync python -m alphaproof.training.rl_cli \
    rl_codet5p_770m_1_8_b200_01 \
    alphaproof/yaml/UCloud/codet5p_770m_1_8_b200.yaml
