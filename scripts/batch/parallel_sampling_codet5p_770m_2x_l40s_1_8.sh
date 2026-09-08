#!/bin/sh
#BSUB -q gpul40s
#BSUB -J parallel_sampling_codet5p_770m_2x_l40s_1_8_01
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o data/runs/parallel_sampling_codet5p_770m_2x_l40s_1_8_01/lsf_%J.out
#BSUB -e data/runs/parallel_sampling_codet5p_770m_2x_l40s_1_8_01/lsf_%J.err

set -eu

cd /work3/s204164/delta-proof
module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

nvidia-smi
uv run --no-sync python -m alphaproof.training.rl_cli \
    parallel_sampling_codet5p_770m_2x_l40s_1_8_01 \
    alphaproof/yaml/codet5p_770m_2x_l40s_1_8_parallel_sampling.yaml
