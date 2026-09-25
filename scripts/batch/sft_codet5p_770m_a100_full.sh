#!/bin/sh
#BSUB -q gpua100
#BSUB -J sft_codet5p_770m_a100_full_01
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -o data/runs/sft_codet5p_770m_a100_full_01/lsf_%J.out
#BSUB -e data/runs/sft_codet5p_770m_a100_full_01/lsf_%J.err

set -eu

cd /work3/s204164/delta-proof
module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

nvidia-smi
uv run --no-sync python -m alphaproof.training.sft \
    sft_codet5p_770m_a100_full_01 \
    alphaproof/yaml/codet5p_770m_a100_full_sft.yaml
