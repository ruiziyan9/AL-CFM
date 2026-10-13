#!/bin/bash
# Pilot: 3 model sizes x short cosine budget on the oracle prior (D=3).
# Submit from the repo root on htc-login:   sbatch CITNP/experiments/arc/pilot_sizes.sh
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=stat-hd-causal
#SBATCH --job-name=alcfm_pilot
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

REPO_DIR=/data/stat-hd-causal/spet5873/AL_CFM
CONDA_ENV=citnp
RESULTS=/data/stat-hd-causal/spet5873/alcfm_results/pilot

SIZES=(s m l)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
EPOCHS=50                 # 50 x 200 = 10k steps
RUN_NAME="pilot_${SIZE}_${SLURM_ARRAY_JOB_ID}"

cd "$REPO_DIR"
module purge; module load Anaconda3; source activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=2 WANDB_MODE=disabled
mkdir -p "$RESULTS"
echo "job $SLURM_JOB_ID size=$SIZE on $(hostname) | git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m CITNP.experiments.mixed_train \
    --data_config  CITNP/experiments/configs/oracle_d3.yaml \
    --model_config CITNP/experiments/configs/oracle_d3_model_${SIZE}.yaml \
    --device cuda --dtype float32 \
    --batch_size 256 --iterations_per_epoch 200 --epochs $EPOCHS \
    --learning_rate 5e-4 --lr_warmup_ratio 0.02 --lr_schedule cosine \
    --save_checkpoint_every_n_steps 2000 \
    --val_iterations 50 --test_iterations 50 \
    --num_workers 4 --log_step 500 --seed 0 --no_wandb \
    --run_name "$RUN_NAME" --results_path "$RESULTS"
