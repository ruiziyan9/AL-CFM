#!/bin/bash
# submit_mixed_smoke.sh — mixed-context MACE-TNP smoke test (D=3) on Oxford ARC/HTC
#
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --job-name=alcfm_smoke_d3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Optional: send email updates
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=ruizi.yan@keble.ox.ac.uk

set -euo pipefail

# ---------------------------------------------------------------- user paths
REPO_DIR="$DATA/AL-CFM"                  # clone of github.com/ruiziyan9/AL-CFM
CONDA_ENV="$DATA/venvs/citnp"             # env built from requirements.txt + pip install -e .
RESULTS_DIR="$DATA/alcfm_results/smoke"  # checkpoints / outputs
# ---------------------------------------------------------------------------

mkdir -p "$REPO_DIR/logs" "$RESULTS_DIR"
cd "$REPO_DIR"

module purge
module load Anaconda3
source activate "$CONDA_ENV"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_MODE=disabled               # belt and braces; we also pass --no_wandb

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== git: $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) uncommitted changes)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "No GPU visible — check --gres / CUDA build of torch"
EOF

# ---- Stage 1: unit tests (model is upstream-exact with empty mask, trainer batch prep)
echo "=== Stage 1: unit tests"
if python -c "import pytest" 2>/dev/null; then
    python -m pytest -q -x \
        testing/test_mixed_context_model.py \
        testing/test_mixed_context_trainer.py
else
    echo "pytest not installed in env — skipping (pip install pytest to enable)"
fi

# ---- Stage 2: built-in smoke check (exits 1 on NaN / no learning)
echo "=== Stage 2: smoke_mixed_d3 (3 epochs x 50 its, batch 16)"
python -m CITNP.experiments.smoke_mixed_d3 \
    --device cuda \
    --num_workers 2

# ---- Stage 3: short end-to-end run through mixed_train (checkpoint + test strata)
echo "=== Stage 3: mixed_train end-to-end"
RUN_NAME="smoke_d3_${SLURM_JOB_ID}"      # must be unique: mixed_train refuses to overwrite
python -m CITNP.experiments.mixed_train \
    --data_config  CITNP/experiments/configs/smoke_d3.yaml \
    --model_config CITNP/experiments/configs/smoke_d3_model.yaml \
    --device cuda \
    --dtype float32 \
    --batch_size 16 \
    --iterations_per_epoch 200 \
    --epochs 10 \
    --val_iterations 20 \
    --test_iterations 20 \
    --num_workers 2 \
    --log_step 50 \
    --seed 0 \
    --no_wandb \
    --run_name "$RUN_NAME" \
    --results_path "$RESULTS_DIR"

echo "=== checkpoints in $RESULTS_DIR/results/$RUN_NAME"
ls -lh "$RESULTS_DIR/results/$RUN_NAME" || true
echo "=== done at $(date)"
