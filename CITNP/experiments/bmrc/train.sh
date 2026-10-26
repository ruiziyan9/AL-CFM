#!/bin/bash
# Parameterised training run on Oxford BMRC (GPU).
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   DATA_CFG=CITNP/experiments/configs/oracle_d3.yaml \
#   MODEL_CFG=CITNP/experiments/configs/oracle_d3_model_m.yaml \
#       sbatch CITNP/experiments/bmrc/train.sh
#
#   DATA_CFG, MODEL_CFG  required
#   RUN         run name (default <data>__<model>__s<SEED>); an existing
#               $OUT/results/$RUN/last_model.pt is resumed, so give every
#               distinct run its own RUN
#   EPOCHS      default 500 (x 200 iterations)
#   SEED        default 0
#   EXTRA_ARGS  further mixed_train arguments
#
# The oracle bank is $OUT/banks/<data>__<sha1 of the data config>.pkl and is
# built when missing, so editing a data config never reuses a stale bank.
# Mail options go on the sbatch command line (--mail-type, --mail-user).
#SBATCH --job-name=alcfm_train
#SBATCH --account=gpu_cebam.prj
#SBATCH --partition=gpu_a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.err
set -euo pipefail
DATA_CFG=${DATA_CFG:?set DATA_CFG, e.g. CITNP/experiments/configs/oracle_d3.yaml}
MODEL_CFG=${MODEL_CFG:?set MODEL_CFG, e.g. CITNP/experiments/configs/oracle_d3_model_m.yaml}
EPOCHS=${EPOCHS:-500}
SEED=${SEED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
RUN=${RUN:-$(basename "$DATA_CFG" .yaml)__$(basename "$MODEL_CFG" .yaml)__s$SEED}
source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

CFG_HASH=$(sha1sum "$DATA_CFG" | cut -c1-10)
BANK=$OUT/banks/$(basename "$DATA_CFG" .yaml)__$CFG_HASH.pkl
mkdir -p "$OUT/banks"
if [ ! -f "$BANK" ]; then
  # Build to a temporary name so a failed or concurrent build never leaves a
  # truncated bank under the final name.
  python -m CITNP.evaluation.oracle_bank \
      --data_config "$DATA_CFG" --out "$BANK.tmp.$$"
  mv "$BANK.tmp.$$" "$BANK"
fi
echo "=== Run $RUN, bank $BANK"

RESUME=""; CKPT=$OUT/results/$RUN/last_model.pt
if [ -f "$CKPT" ]; then RESUME="--resume_full $CKPT"; echo "=== Resuming from $CKPT"; fi
# shellcheck disable=SC2086  # EXTRA_ARGS and RESUME are word-split on purpose
python -u -m CITNP.experiments.mixed_train \
  --data_config "$DATA_CFG" \
  --model_config "$MODEL_CFG" \
  --device cuda --dtype float32 \
  --batch_size 256 --iterations_per_epoch 200 --epochs "$EPOCHS" \
  --learning_rate 5e-4 --lr_warmup_ratio 0.02 --lr_schedule cosine \
  --save_checkpoint_every_n_steps 2500 --val_iterations 50 --test_iterations 50 \
  --num_workers 4 --log_step 500 --seed "$SEED" --no_wandb \
  --oracle_bank "$BANK" --oracle_every 2500 \
  --run_name "$RUN" --results_path "$OUT" $RESUME $EXTRA_ARGS
