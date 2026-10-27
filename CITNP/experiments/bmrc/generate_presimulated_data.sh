#!/bin/bash
# Generate one split of a pre-simulated dataset on Oxford BMRC (CPU array).
#
# Job i of NUM_JOBS writes chunk files i, i + NUM_JOBS, i + 2 * NUM_JOBS, ...,
# so the fixed array below covers any number of files, and chunk files that
# already exist are skipped: resubmitting after a failure only fills the gaps.
# Options given on the sbatch command line override the #SBATCH defaults, e.g.
#   sbatch --array=0-9 --time=01:00:00 ...   (then also set NUM_JOBS=10)
#
#   DATASET_CONFIG  required, e.g.
#               CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml
#   SPLIT       required: train, validation or test
#   NUM_JOBS    number of array jobs (default 50; must equal the array size)
#   DATA_ROOT   where datasets are written (default $OUT/data, outside the checkout)
#   DRY_RUN=1   print the resolved variables and the command, then exit. Loads
#               no module, activates no conda env and imports no Python, so it
#               runs with plain `bash` on the login node.
#
# Submit all three splits:
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   export DATASET_CONFIG=CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml
#   SPLIT=train      sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
#   SPLIT=validation sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
#   SPLIT=test       sbatch CITNP/experiments/bmrc/generate_presimulated_data.sh
#SBATCH --job-name=alcfm_generate
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%A_%a.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%A_%a.err
set -euo pipefail
DATASET_CONFIG=${DATASET_CONFIG:?set DATASET_CONFIG, e.g. CITNP/experiments/configs/datasets/oracle_d3_iw_motif__v1.yaml}
SPLIT=${SPLIT:?set SPLIT to train, validation or test}
NUM_JOBS=${NUM_JOBS:-50}
DRY_RUN=${DRY_RUN:-0}
JOB_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Same defaults as common_env.sh, resolved here so DRY_RUN needs no module.
USER=${USER:-$(id -un)}
W=${W:-/gpfs3/well/cebam/users/$USER}
REPO=${REPO:-$W/AL-CFM}
OUT=${OUT:-$W/alcfm}
DATA_ROOT=${DATA_ROOT:-$OUT/data}
if [ -d "$REPO" ]; then cd "$REPO"; fi

GENERATE_CMD=(python -u -m CITNP.datasets.create_mixed_regime_chunks
  --dataset_config "$DATASET_CONFIG" --split "$SPLIT" --data_root "$DATA_ROOT"
  --job_index "$JOB_INDEX" --num_jobs "$NUM_JOBS" --skip_existing_files)

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: nothing is loaded or generated"
  echo "REPO=$REPO"
  echo "DATASET_CONFIG=$DATASET_CONFIG"
  echo "SPLIT=$SPLIT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "NUM_JOBS=$NUM_JOBS (JOB_INDEX=$JOB_INDEX; under sbatch: \$SLURM_ARRAY_TASK_ID)"
  echo "${GENERATE_CMD[*]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" != "$NUM_JOBS" ]; then
  echo "ABORT: the array has $SLURM_ARRAY_TASK_COUNT jobs but NUM_JOBS=$NUM_JOBS" >&2
  exit 1
fi
"${GENERATE_CMD[@]}"
