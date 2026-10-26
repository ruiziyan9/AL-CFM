#!/bin/bash
# Parameterised training run on Oxford BMRC (GPU).
#
# Options given on the sbatch command line override the #SBATCH defaults
# below, e.g. another partition or a shorter time limit:
#   sbatch --partition=gpu_short --time=01:00:00 CITNP/experiments/bmrc/train.sh
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   DATA_CFG=CITNP/experiments/configs/oracle_d3.yaml \
#   MODEL_CFG=CITNP/experiments/configs/oracle_d3_model_m.yaml \
#       sbatch CITNP/experiments/bmrc/train.sh
#
#   DATA_CFG, MODEL_CFG  required
#   RUN         run name (default <data>__<model>__s<SEED>); never reuse a
#               name for a different experiment (see the resume guard below)
#   EPOCHS      default 500 (x 200 iterations)
#   SEED        default 0
#   EXTRA_ARGS  further mixed_train arguments
#   BANK_ARGS   further arguments of the oracle-bank build (e.g. --per_bucket 5)
#   DRY_RUN=1   print the resolved variables and the exact commands, then
#               exit. Loads no module, activates no conda env, imports no
#               Python and needs no GPU, so it runs with plain `bash` on the
#               login node:  DRY_RUN=1 DATA_CFG=... MODEL_CFG=... \
#                                bash CITNP/experiments/bmrc/train.sh
#   FORCE_RESUME=1  resume a run that has no recorded config hashes (a run
#               started before this script existed), then record the current
#               hashes. A hash mismatch is never overridden.
#
# The oracle bank is $OUT/banks/<data>__<sha1 of the data config>.pkl and is
# built when missing, so editing a data config never reuses a stale bank.
# Resume guard: the first launch of a RUN writes the sha1 of its data and
# model configs to $OUT/results/$RUN/config_hashes.txt. If last_model.pt
# exists the run resumes only when both hashes still match; otherwise the
# script aborts. Mail options go on the sbatch command line.
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
BANK_ARGS=${BANK_ARGS:-}
DRY_RUN=${DRY_RUN:-0}
FORCE_RESUME=${FORCE_RESUME:-0}
RUN=${RUN:-$(basename "$DATA_CFG" .yaml)__$(basename "$MODEL_CFG" .yaml)__s$SEED}

# Same defaults as common_env.sh, resolved here so DRY_RUN needs no module.
USER=${USER:-$(id -un)}
W=${W:-/gpfs3/well/cebam/users/$USER}
REPO=${REPO:-$W/AL-CFM}
OUT=${OUT:-$W/alcfm}
if [ -d "$REPO" ]; then cd "$REPO"; fi

hash10() { sha1sum "$1" | cut -c1-10; }
DATA_HASH=$(hash10 "$DATA_CFG")
MODEL_HASH=$(hash10 "$MODEL_CFG")
BANK=$OUT/banks/$(basename "$DATA_CFG" .yaml)__$DATA_HASH.pkl
RUN_DIR=$OUT/results/$RUN
CKPT=$RUN_DIR/last_model.pt
HASHES=$RUN_DIR/config_hashes.txt
WANT_HASHES="data=$DATA_HASH
model=$MODEL_HASH"

# Resume decision: RESUME is "" or "--resume_full <ckpt>"; abort on mismatch.
RESUME=""
RECORD_HASHES=0
if [ -f "$CKPT" ]; then
  if [ -f "$HASHES" ]; then
    if [ "$(cat "$HASHES")" != "$WANT_HASHES" ]; then
      echo "ABORT: run '$RUN' has a checkpoint recorded with different configs." >&2
      echo "  recorded ($HASHES):" >&2
      sed 's/^/    /' "$HASHES" >&2
      echo "  current:" >&2
      echo "$WANT_HASHES" | sed 's/^/    /' >&2
      echo "  Use a new RUN name, or restore the original configs." >&2
      exit 1
    fi
    RESUME="--resume_full $CKPT"
  elif [ "$FORCE_RESUME" = 1 ]; then
    RESUME="--resume_full $CKPT"
    RECORD_HASHES=1
  else
    echo "ABORT: run '$RUN' has a checkpoint but no $HASHES," >&2
    echo "  so its configs cannot be verified. Use a new RUN name, or set" >&2
    echo "  FORCE_RESUME=1 if you know the current configs are the original ones." >&2
    exit 1
  fi
else
  RECORD_HASHES=1
fi

read -ra EXTRA <<< "$EXTRA_ARGS"
read -ra BANK_EXTRA <<< "$BANK_ARGS"
read -ra RESUME_ARGS <<< "$RESUME"
BANK_CMD=(python -m CITNP.evaluation.oracle_bank
  --data_config "$DATA_CFG" --out "$BANK.tmp.PID" ${BANK_EXTRA[@]+"${BANK_EXTRA[@]}"})
TRAIN_CMD=(python -u -m CITNP.experiments.mixed_train
  --data_config "$DATA_CFG"
  --model_config "$MODEL_CFG"
  --device cuda --dtype float32
  --batch_size 256 --iterations_per_epoch 200 --epochs "$EPOCHS"
  --learning_rate 5e-4 --lr_warmup_ratio 0.02 --lr_schedule cosine
  --save_checkpoint_every_n_steps 2500 --val_iterations 50 --test_iterations 50
  --num_workers 4 --log_step 500 --seed "$SEED" --no_wandb
  --oracle_bank "$BANK" --oracle_every 2500
  --run_name "$RUN" --results_path "$OUT"
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})

# One "--flag value" per line, continued with a backslash.
print_cmd() {
  local line="$1" a; shift
  for a in "$@"; do
    case $a in
      -*) printf '%s \\\n' "$line"; line="  $a" ;;
      *) line="$line $a" ;;
    esac
  done
  printf '%s\n' "$line"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: nothing is loaded, built or launched"
  echo "REPO=$REPO"
  echo "OUT=$OUT"
  echo "RUN=$RUN"
  echo "DATA_CFG=$DATA_CFG (sha1 $DATA_HASH)"
  echo "MODEL_CFG=$MODEL_CFG (sha1 $MODEL_HASH)"
  echo "BANK=$BANK"
  if [ -f "$BANK" ]; then echo "# bank exists: not rebuilt"; else
    echo "# bank missing: built first with ('PID' is the job's process id)"
    print_cmd "${BANK_CMD[@]}"
  fi
  if [ -n "$RESUME" ] && [ "$RECORD_HASHES" = 1 ]; then
    echo "# resume: yes, FORCE_RESUME=1 (current hashes recorded to $HASHES)"
  elif [ -n "$RESUME" ]; then echo "# resume: yes (configs match $HASHES)"
  else echo "# resume: no (fresh run; hashes written to $HASHES)"; fi
  print_cmd "${TRAIN_CMD[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

mkdir -p "$OUT/banks" "$RUN_DIR"
if [ ! -f "$BANK" ]; then
  # Build to a temporary name so a failed or concurrent build never leaves a
  # truncated bank under the final name.
  python -m CITNP.evaluation.oracle_bank \
      --data_config "$DATA_CFG" --out "$BANK.tmp.$$" ${BANK_EXTRA[@]+"${BANK_EXTRA[@]}"}
  mv "$BANK.tmp.$$" "$BANK"
fi
if [ "$RECORD_HASHES" = 1 ]; then echo "$WANT_HASHES" > "$HASHES"; fi
echo "=== Run $RUN, bank $BANK"
if [ -n "$RESUME" ]; then echo "=== Resuming from $CKPT"; fi
"${TRAIN_CMD[@]}"
