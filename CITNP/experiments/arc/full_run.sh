#!/bin/bash
# Full training run on the oracle prior (D=3). Set SIZE / EPOCHS from the pilot.
#   sbatch CITNP/experiments/arc/full_run.sh
#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --account=stat-hd-causal
#SBATCH --job-name=alcfm_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO_DIR=/data/stat-hd-causal/spet5873/AL_CFM
CONDA_ENV=/data/stat-hd-causal/spet5873/envs/citnp
RESULTS=/data/stat-hd-causal/spet5873/alcfm_results/full

SIZE=l          # from the pilot
EPOCHS=500      # from the pilot: 500 x 200 = 100k steps; must fit in --time
RUN_NAME="oracle_d3_${SIZE}_e${EPOCHS}_${SLURM_JOB_ID}"

cd "$REPO_DIR"
module purge; module load Anaconda3; source activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=2 WANDB_MODE=disabled
mkdir -p "$RESULTS"
echo "job $SLURM_JOB_ID size=$SIZE epochs=$EPOCHS on $(hostname) | git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$CONDA_ENV"/bin/python -m CITNP.experiments.mixed_train \
    --data_config  CITNP/experiments/configs/oracle_d3.yaml \
    --model_config CITNP/experiments/configs/oracle_d3_model_${SIZE}.yaml \
    --device cuda --dtype float32 \
    --batch_size 256 --iterations_per_epoch 200 --epochs $EPOCHS \
    --learning_rate 5e-4 --lr_warmup_ratio 0.02 --lr_schedule cosine \
    --save_checkpoint_every_n_steps 10000 \
    --val_iterations 50 --test_iterations 50 \
    --num_workers 4 --log_step 500 --seed 0 --no_wandb \
    --run_name "$RUN_NAME" --results_path "$RESULTS"
