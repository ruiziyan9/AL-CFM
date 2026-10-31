#!/bin/bash
# kshot-prior evaluation as a Slurm array over task_offset (CPU by default:
# the exact oracle dominates the cost). Array task a evaluates tasks
# a * TASKS_PER_JOB .. (a + 1) * TASKS_PER_JOB - 1; task keys depend only on
# (D, task index), so the split does not change any result.
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   REPO=$PWD CKPT=... MODEL_CFG=... DATA_CFG=... OUT_PREFIX=... \
#       sbatch --array=0-9 CITNP/experiments/bmrc/eval_kshot_prior.sh
#   # then, once all parts have finished:
#   REPO=$PWD OUT_PREFIX=... \
#       sbatch --dependency=afterok:<array job id> \
#       CITNP/experiments/bmrc/merge_kshot_prior.sh
#
#   CKPT, MODEL_CFG, DATA_CFG  required
#   OUT_PREFIX     required; part a writes ${OUT_PREFIX}_part<a>.csv,
#                  _part<a>_by_t.csv and _part<a>_meta.json
#   TASKS_PER_JOB  default 50
#   K_LIST         default 0,1,3,5,10
#   N_MC           default 2000
#   NUM_VARIABLES  optional comma-separated D values (default: all in DATA_CFG)
#   DEVICE         default cpu. For a GPU add to the sbatch command line
#                  --account=gpu_cebam.prj --partition=gpu_a100_80gb
#                  --gres=gpu:1 and set DEVICE=cuda.
#   EXTRA_ARGS     further kshot-prior arguments (e.g. "--n_y 32")
#   ARRAY_ID       array index when not run through Slurm (default 0), e.g.
#                  in an interactive `srun --pty bash` session:
#                  ARRAY_ID=1 bash CITNP/experiments/bmrc/eval_kshot_prior.sh
#   DRY_RUN=1      print the resolved command and exit (plain `bash` works)
#
# Adjust --array and --time to the number of tasks and the pilot's wall time.
#SBATCH --job-name=alcfm_kshot_prior
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%A_%a.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%A_%a.err
set -euo pipefail
CKPT=${CKPT:?set CKPT}
MODEL_CFG=${MODEL_CFG:?set MODEL_CFG}
DATA_CFG=${DATA_CFG:?set DATA_CFG}
OUT_PREFIX=${OUT_PREFIX:?set OUT_PREFIX}
TASKS_PER_JOB=${TASKS_PER_JOB:-50}
K_LIST=${K_LIST:-0,1,3,5,10}
N_MC=${N_MC:-2000}
NUM_VARIABLES=${NUM_VARIABLES:-}
DEVICE=${DEVICE:-cpu}
EXTRA_ARGS=${EXTRA_ARGS:-}
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-${ARRAY_ID:-0}}
DRY_RUN=${DRY_RUN:-0}
OFFSET=$((ARRAY_ID * TASKS_PER_JOB))
PART=${OUT_PREFIX}_part${ARRAY_ID}

read -ra EXTRA <<< "$EXTRA_ARGS"
NV=()
if [ -n "$NUM_VARIABLES" ]; then NV=(--num_variables "$NUM_VARIABLES"); fi
CMD=(python -u -m CITNP.evaluation kshot-prior
  --ckpt "$CKPT" --model_config "$MODEL_CFG" --data_config "$DATA_CFG"
  --n_tasks "$TASKS_PER_JOB" --task_offset "$OFFSET"
  --k_list "$K_LIST" --n_mc "$N_MC" --device "$DEVICE"
  ${NV[@]+"${NV[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
  --out "$PART.csv")

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: array index $ARRAY_ID, tasks $OFFSET..$((OFFSET + TASKS_PER_JOB - 1))"
  echo "${CMD[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
mkdir -p "$(dirname "$OUT_PREFIX")"
echo "=== kshot-prior part $ARRAY_ID: tasks $OFFSET..$((OFFSET + TASKS_PER_JOB - 1))"
start=$(date +%s)
"${CMD[@]}"
echo "=== wall time $(($(date +%s) - start)) s for $TASKS_PER_JOB tasks"
