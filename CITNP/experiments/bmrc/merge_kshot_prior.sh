#!/bin/bash
# Merge the parts written by eval_kshot_prior.sh and draw the figures
# (plot-kshot-prior refuses parts whose data-config hashes differ). Submit
# after the array, or run it in an interactive session:
#
#   REPO=$PWD OUT_PREFIX=... \
#       sbatch --dependency=afterok:<array job id> \
#       CITNP/experiments/bmrc/merge_kshot_prior.sh
#
#   OUT_PREFIX  required; reads ${OUT_PREFIX}_part*.csv, writes the figures
#               ${OUT_PREFIX}_D<D>_main.png and _D<D>_by_relation.png
#   DRY_RUN=1   print the resolved command and exit
#SBATCH --job-name=alcfm_kshot_prior_merge
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.err
set -euo pipefail
OUT_PREFIX=${OUT_PREFIX:?set OUT_PREFIX}
DRY_RUN=${DRY_RUN:-0}

PARTS=()
for f in "${OUT_PREFIX}"_part*.csv; do
  case $f in *_by_t.csv) ;; *) [ -e "$f" ] && PARTS+=("$f") ;; esac
done
if [ ${#PARTS[@]} -eq 0 ]; then
  echo "ABORT: no ${OUT_PREFIX}_part*.csv found" >&2
  exit 1
fi
CMD=(python -m CITNP.evaluation plot-kshot-prior "${PARTS[@]}" --prefix "$OUT_PREFIX")

if [ "$DRY_RUN" = 1 ]; then
  echo "${CMD[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
echo "=== merging ${#PARTS[@]} parts"
"${CMD[@]}"
