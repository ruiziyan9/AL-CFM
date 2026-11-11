#!/bin/bash
# kshot-prior evaluation as a Slurm array over task_offset (CPU by default:
# the exact oracle dominates the cost; see NO_ORACLE for the model alone).
# Array task a evaluates tasks
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
#   ORACLE_BUDGET  total (default: N_MC oracle draws per query in total) or
#                  per_dag (N_MC per retained DAG); --oracle_budget
#   NO_ORACLE=1    evaluate the model against the truth only (--no_oracle):
#                  the exact oracle is not run and its columns are left
#                  empty. Needed for D above the oracle's limit (five
#                  variables); the model scores equal those of a run with
#                  the oracle. N_MC and ORACLE_BUDGET have no effect in this
#                  mode. Without the oracle the run is cheap: one GPU job
#                  replaces the CPU array, e.g.
#                    NO_ORACLE=1 DEVICE=cuda TASKS_PER_JOB=500 ... \
#                        sbatch --array=0 --account=gpu_cebam.prj \
#                        --partition=gpu_a100_80gb --gres=gpu:1 \
#                        CITNP/experiments/bmrc/eval_kshot_prior.sh
#                  Parts run with and without the oracle cannot be merged.
#   FIXED_GRAPH_EDGES  evaluate on ONE given DAG instead of graphs drawn from
#                  the training prior (--fixed_graph_edges): 0-based
#                  "source->target" edges, node j being variable X_{j+1}.
#                  Weights, noise and do-values are still drawn from DATA_CFG
#                  as in training. Needs a single D (NUM_VARIABLES, unless
#                  DATA_CFG has one). With the oracle (D <= 4) it is no longer
#                  a guaranteed lower bound. Example, five variables:
#                    FIXED_GRAPH_EDGES="0->1,1->2,2->4,0->3,3->4,0->2" \
#                        NO_ORACLE=1 ...
#                  Parts with different graphs, or with and without one,
#                  cannot be merged.
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
# Threads: common_env.sh pins OMP/MKL to 1 thread; this script then sets
# OMP_NUM_THREADS and MKL_NUM_THREADS to SLURM_CPUS_PER_TASK (1 outside
# Slurm), and kshot-prior calls torch.set_num_threads(OMP_NUM_THREADS).
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
ORACLE_BUDGET=${ORACLE_BUDGET:-total}
NO_ORACLE=${NO_ORACLE:-0}
FIXED_GRAPH_EDGES=${FIXED_GRAPH_EDGES:-}
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
ORACLE=()
if [ "$NO_ORACLE" = 1 ]; then ORACLE=(--no_oracle); fi
FIXED_GRAPH=()
if [ -n "$FIXED_GRAPH_EDGES" ]; then
  FIXED_GRAPH=(--fixed_graph_edges "$FIXED_GRAPH_EDGES")
fi
CMD=(python -u -m CITNP.evaluation kshot-prior
  --ckpt "$CKPT" --model_config "$MODEL_CFG" --data_config "$DATA_CFG"
  --n_tasks "$TASKS_PER_JOB" --task_offset "$OFFSET"
  --k_list "$K_LIST" --n_mc "$N_MC" --oracle_budget "$ORACLE_BUDGET"
  --device "$DEVICE"
  ${ORACLE[@]+"${ORACLE[@]}"}
  ${FIXED_GRAPH[@]+"${FIXED_GRAPH[@]}"}
  ${NV[@]+"${NV[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
  --out "$PART.csv")

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: array index $ARRAY_ID, tasks $OFFSET..$((OFFSET + TASKS_PER_JOB - 1))"
  # the edges are shown in quotes: unquoted, the ">" would redirect output
  shown=("${CMD[@]}")
  for i in "${!shown[@]}"; do
    if [ -n "$FIXED_GRAPH_EDGES" ] && [ "${shown[$i]}" = "$FIXED_GRAPH_EDGES" ]; then
      shown[$i]="\"$FIXED_GRAPH_EDGES\""
    fi
  done
  echo "${shown[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} \
  MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$(dirname "$OUT_PREFIX")"
echo "=== kshot-prior part $ARRAY_ID: tasks $OFFSET..$((OFFSET + TASKS_PER_JOB - 1))"
start=$(date +%s)
"${CMD[@]}"
echo "=== wall time $(($(date +%s) - start)) s for $TASKS_PER_JOB tasks"
