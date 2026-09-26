#!/bin/sh
#BSUB -q gpua100
#BSUB -J eval_sft_l40s_1_2_rl_train_1000
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -M 20GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o data/evaluations/sft_codet5p_770m_l40s_1_2_rl_train_1000_seed42/lsf_%J.out
#BSUB -e data/evaluations/sft_codet5p_770m_l40s_1_2_rl_train_1000_seed42/lsf_%J.err

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

experiment_dir=data/evaluations/sft_codet5p_770m_l40s_1_2_rl_train_1000_seed42
batch_id=sft_codet5p_770m_l40s_1_2_rl_train_1000_seed42

nvidia-smi
uv run --no-sync python scripts/prepare_evaluation_requests.py \
    --input data/dataset/numina_math_lean_passing_1_8/train.jsonl \
    --output "$experiment_dir/requests.jsonl" \
    --sample-size 1000 \
    --seed 42 \
    --batch-id "$batch_id"

uv run --no-sync python -m alphaproof.inference.infer \
    --config alphaproof/yaml/codet5p_770m_l40s_1_8.yaml \
    --input "$experiment_dir/requests.jsonl" \
    --batch-id "$batch_id" \
    --run-dir data/runs/sft_codet5p_770m_l40s_1_2_01 \
    --num-simulations 64 \
    --num-sampled-actions 6 \
    --tactic-timeout 1.0 \
    --parallel-searches 96 \
    --max-concurrent-lean-imports 24 \
    --inference-num-gpus 1 \
    --inference-batch-size 30 \
    --inference-batch-timeout 0.05 \
    --final-check-timeout 300 \
    --seed 42 \
    > "$experiment_dir/results.jsonl"

uv run --no-sync python scripts/summarize_inference_results.py \
    --input "$experiment_dir/results.jsonl" \
    --output "$experiment_dir/summary.json" \
    --expected-results 1000
