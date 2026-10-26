#!/bin/bash
# Submit the 2 x 2 information-parity grid (4 GPU jobs via train.sh).
#
#   data prior \ model     no scale input     scale_conditioning: stats
#   IW motif (eta = 1)     A                  B
#   NIG eta = 9 motif      C                  D
#
# Run from the repo root on the login node:
#   bash CITNP/experiments/bmrc/submit_parity_grid.sh
#   DRY_RUN=1 bash CITNP/experiments/bmrc/submit_parity_grid.sh   # print only
#
# Same SEED and EPOCHS for all four runs. Every run has its own RUN name, so
# train.sh's auto-resume can never pick up another run's checkpoint; rerunning
# this script after an interruption resumes each run (same configs and EPOCHS).
#   ENC     r1 | layerwise | extra_node (default extra_node)
#   SEED    default 0
#   EPOCHS  default 500
#   TAG     run-name prefix (default parity)
# Mail options and other sbatch flags go in SBATCH_ARGS.
# Both IW-motif runs (A, B) share one oracle bank, and so do C and D; if a bank
# is missing, build it once first (see the hand-off) so the two jobs do not race.
set -euo pipefail
ENC=${ENC:-extra_node}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-500}
TAG=${TAG:-parity}
DRY_RUN=${DRY_RUN:-0}
SBATCH_ARGS=${SBATCH_ARGS:-}
CFG=CITNP/experiments/configs
USER=${USER:-$(id -un)}
W=${W:-/gpfs3/well/cebam/users/$USER}
OUT=${OUT:-$W/alcfm}

case $ENC in
  extra_node) MODEL=oracle_d3_model_m_extra_node ;;
  r1)         MODEL=oracle_d3_model_m ;;
  layerwise)  MODEL=oracle_d3_model_m_layerwise ;;
  *) echo "unknown ENC=$ENC (r1, layerwise, extra_node)" >&2; exit 1 ;;
esac
IW=$CFG/oracle_d3_iw_motif.yaml
NIG=$CFG/oracle_d3_nig_eta9_motif.yaml

submit() { # name data_cfg model_cfg
  local run="${TAG}_$1_${ENC}_s${SEED}"
  [ -f "$2" ] && [ -f "$3" ] || { echo "missing config: $2 or $3" >&2; exit 1; }
  local cmd=(sbatch --job-name="$run" $SBATCH_ARGS
    --export=ALL,DATA_CFG="$2",MODEL_CFG="$3",RUN="$run",EPOCHS="$EPOCHS",SEED="$SEED"
    CITNP/experiments/bmrc/train.sh)
  if [ "$DRY_RUN" = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

[ "$DRY_RUN" = 1 ] || mkdir -p "$OUT/logs"
submit A "$IW"  "$CFG/$MODEL.yaml"
submit B "$IW"  "$CFG/${MODEL}_stats.yaml"
submit C "$NIG" "$CFG/$MODEL.yaml"
submit D "$NIG" "$CFG/${MODEL}_stats.yaml"
