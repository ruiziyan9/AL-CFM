#!/bin/bash
# Intervention-encoding diagnostics of one trained checkpoint (CPU by
# default): embedding-geometry (extra_node models only), probe-intervention
# and context-edits; see CITNP/evaluation/INTERVENTION_DIAGNOSTICS_README.md.
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   REPO=$PWD CKPT=<run dir>/best_model.pt \
#       MODEL_CFG=<run dir>/resolved_model_config.yaml \
#       DATA_CFG=CITNP/experiments/configs/oracle_d3_iw_motif.yaml \
#       OUT_PREFIX=<results dir>/extra_node_m \
#       sbatch CITNP/experiments/bmrc/eval_intervention_diagnostics.sh
#
#   CKPT, MODEL_CFG, DATA_CFG  required. DATA_CFG is the prior config the
#                  model was trained on, or a pre-simulated dataset config
#                  (configs/datasets/*.yaml; its prior_config is used, the
#                  chunk files are not read). context-edits needs a
#                  linear-Gaussian prior with normalise: false.
#   OUT_PREFIX     required; writes ${OUT_PREFIX}_metadata_embedding_geometry.json,
#                  ${OUT_PREFIX}_intervention_probes{.csv,_meta.json} and two
#                  context-edits runs, each {.csv,_summary.csv,_meta.json}:
#                  ${OUT_PREFIX}_context_edits_original_normalisation
#                    (--keep_original_normalisation: only the mask changes;
#                    the primary readout) and
#                  ${OUT_PREFIX}_context_edits_edited_normalisation
#                    (the model's normal input path).
#                  Keep it outside the git checkout.
#   DIAGNOSTICS    default "embedding-geometry probe-intervention
#                  context-edits"; embedding-geometry is skipped unless
#                  MODEL_CFG selects extra_node
#   N_TASKS        default 200 (both probe-intervention and context-edits)
#   TASK_OFFSET    default 0
#   NUM_VARIABLES  optional D (default: the smallest D of the prior)
#   MAX_BATCH_SIZE optional cap on (context version, query) elements per
#                  model forward pass in context-edits (default: one pass
#                  per context; set it if memory runs out)
#   DEVICE         default cpu (see eval_kshot_prior.sh for GPU options)
#   PROBE_ARGS     further probe-intervention arguments (e.g. "--rows_per_task 64")
#   EDIT_ARGS      further context-edits arguments (e.g. "--n_mc 1000")
#   DRY_RUN=1      print the resolved commands and exit (plain `bash` works)
#
# Threads: as eval_kshot_prior.sh (OMP / MKL threads = SLURM_CPUS_PER_TASK).
#SBATCH --job-name=alcfm_intervention_diagnostics
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.err
set -euo pipefail
CKPT=${CKPT:?set CKPT}
MODEL_CFG=${MODEL_CFG:?set MODEL_CFG}
DATA_CFG=${DATA_CFG:?set DATA_CFG}
OUT_PREFIX=${OUT_PREFIX:?set OUT_PREFIX}
DIAGNOSTICS=${DIAGNOSTICS:-embedding-geometry probe-intervention context-edits}
N_TASKS=${N_TASKS:-200}
TASK_OFFSET=${TASK_OFFSET:-0}
NUM_VARIABLES=${NUM_VARIABLES:-}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-}
DEVICE=${DEVICE:-cpu}
PROBE_ARGS=${PROBE_ARGS:-}
EDIT_ARGS=${EDIT_ARGS:-}
DRY_RUN=${DRY_RUN:-0}

read -ra PROBE_EXTRA <<< "$PROBE_ARGS"
read -ra EDIT_EXTRA <<< "$EDIT_ARGS"
NV=()
if [ -n "$NUM_VARIABLES" ]; then NV=(--num_variables "$NUM_VARIABLES"); fi
BATCH=()
if [ -n "$MAX_BATCH_SIZE" ]; then BATCH=(--max_batch_size "$MAX_BATCH_SIZE"); fi
COMMON=(--ckpt "$CKPT" --model_config "$MODEL_CFG" --device "$DEVICE")
TASKS=(--data_config "$DATA_CFG" --n_tasks "$N_TASKS"
  --task_offset "$TASK_OFFSET" ${NV[@]+"${NV[@]}"})

if grep -Eq '^(intervention_encoding:[[:space:]]*extra_node|model_class:[[:space:]]*ExtraNode)' \
    "$MODEL_CFG"; then
  EXTRA_NODE=1
else
  EXTRA_NODE=0
fi

commands=()
for diagnostic in $DIAGNOSTICS; do
  case "$diagnostic" in
    embedding-geometry)
      if [ "$EXTRA_NODE" = 0 ]; then
        echo "=== skipping embedding-geometry: $MODEL_CFG is not extra_node"
        continue
      fi
      GEOMETRY_NV=()
      if [ -n "$NUM_VARIABLES" ]; then
        GEOMETRY_NV=(--num_variables "$NUM_VARIABLES")
      fi
      commands+=("python -u -m CITNP.evaluation embedding-geometry $(printf '%q ' \
        "${COMMON[@]}" ${GEOMETRY_NV[@]+"${GEOMETRY_NV[@]}"} \
        --out "${OUT_PREFIX}_metadata_embedding_geometry.json")")
      ;;
    probe-intervention)
      commands+=("python -u -m CITNP.evaluation probe-intervention $(printf '%q ' \
        "${COMMON[@]}" "${TASKS[@]}" ${PROBE_EXTRA[@]+"${PROBE_EXTRA[@]}"} \
        --out "${OUT_PREFIX}_intervention_probes.csv")")
      ;;
    context-edits)
      # primary readout first: only the intervention mask changes
      commands+=("python -u -m CITNP.evaluation context-edits $(printf '%q ' \
        "${COMMON[@]}" "${TASKS[@]}" ${BATCH[@]+"${BATCH[@]}"} \
        ${EDIT_EXTRA[@]+"${EDIT_EXTRA[@]}"} --keep_original_normalisation \
        --out "${OUT_PREFIX}_context_edits_original_normalisation.csv")")
      commands+=("python -u -m CITNP.evaluation context-edits $(printf '%q ' \
        "${COMMON[@]}" "${TASKS[@]}" ${BATCH[@]+"${BATCH[@]}"} \
        ${EDIT_EXTRA[@]+"${EDIT_EXTRA[@]}"} \
        --out "${OUT_PREFIX}_context_edits_edited_normalisation.csv")")
      ;;
    *)
      echo "unknown diagnostic: $diagnostic" >&2
      exit 2
      ;;
  esac
done

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN"
  printf '%s\n' "${commands[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} \
  MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$(dirname "$OUT_PREFIX")"
for command in "${commands[@]}"; do
  echo "=== $command"
  start=$(date +%s)
  eval "$command"
  echo "=== wall time $(($(date +%s) - start)) s"
done
