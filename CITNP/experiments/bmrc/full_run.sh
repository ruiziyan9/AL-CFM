#!/bin/bash
# Full training run on Oxford BMRC. Data and model configs are parameters:
#
#   mkdir -p logs
#   sbatch --export=ALL,DATA_CFG=CITNP/experiments/configs/oracle_d3_strong.yaml,SIZE=m \
#       CITNP/experiments/bmrc/full_run.sh
#
# Optional: EPOCHS (default 500), MODEL_CFG (default oracle_d3_model_${SIZE}.yaml),
# ORACLE_BANK (bank.pkl from `python -m CITNP.evaluation bank`) to log the
# model-oracle gap during training. Check the partition name with `sinfo -s`.
#SBATCH --partition=gpu_long
#SBATCH --account=gpu_cebam.prj
#SBATCH --job-name=alcfm_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO_DIR=${REPO_DIR:-/gpfs3/well/cebam/users/$USER/AL-CFM}
RESULTS=${RESULTS:-/gpfs3/well/cebam/users/$USER/alcfm_results/full}
CONDA_ENV=${CONDA_ENV:-mace_tnp}
DATA_CFG=${DATA_CFG:?set DATA_CFG, e.g. CITNP/experiments/configs/oracle_d3_strong.yaml}
SIZE=${SIZE:-m}
MODEL_CFG=${MODEL_CFG:-CITNP/experiments/configs/oracle_d3_model_${SIZE}.yaml}
EPOCHS=${EPOCHS:-500}   # 500 x 200 = 100k steps; must fit in --time
RUN_NAME="$(basename "$DATA_CFG" .yaml)_${SIZE}_e${EPOCHS}_${SLURM_JOB_ID}"

cd "$REPO_DIR"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=2 WANDB_MODE=disabled
mkdir -p "$RESULTS"
echo "job $SLURM_JOB_ID | data=$DATA_CFG model=$MODEL_CFG epochs=$EPOCHS | $(hostname) | git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA=()
if [[ -n "${ORACLE_BANK:-}" ]]; then
    EXTRA+=(--oracle_bank "$ORACLE_BANK")
fi

python -m CITNP.experiments.mixed_train \
    --data_config "$DATA_CFG" \
    --model_config "$MODEL_CFG" \
    --device cuda --dtype float32 \
    --batch_size 256 --iterations_per_epoch 200 --epochs "$EPOCHS" \
    --learning_rate 5e-4 --lr_warmup_ratio 0.02 --lr_schedule cosine \
    --save_checkpoint_every_n_steps 10000 \
    --val_iterations 50 --test_iterations 50 \
    --num_workers 4 --log_step 500 --seed 0 --no_wandb \
    --run_name "$RUN_NAME" --results_path "$RESULTS" \
    "${EXTRA[@]}"
