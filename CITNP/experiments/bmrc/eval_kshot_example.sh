#!/bin/bash
# kshot-example: one fixed model and one query evaluated over repeated context
# draws (CPU by default; the exact oracle dominates the cost, so use NO_ORACLE
# above its variable limit). A single job, no array -- one worked example.
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   REPO=$PWD CKPT=... MODEL_CFG=... DATA_CFG=... OUT_PREFIX=... \
#       TREATMENT=2 OUTCOME=4 FIXED_GRAPH_EDGES="0->1,1->2,2->4,0->3,3->4,0->2" \
#       NO_ORACLE=1 sbatch CITNP/experiments/bmrc/eval_kshot_example.sh
#
#   CKPT, MODEL_CFG     required
#   OUT_PREFIX          required; writes ${OUT_PREFIX}.csv, _by_t.csv,
#                       _meta.json and the figures _kl.png and _effect.png
#   TREATMENT, OUTCOME  required; 0-based treatment T and outcome Y of the
#                       query X_{Y+1} | do(X_{T+1} = t)
#   The model comes from EXAMPLE_FILE, or, when that is empty, is drawn from
#   DATA_CFG's prior on FIXED_GRAPH_EDGES (and saved next to OUT_PREFIX):
#   EXAMPLE_FILE        a YAML example model (overrides FIXED_GRAPH_EDGES)
#   FIXED_GRAPH_EDGES   0-based "source->target" edges, node j being X_{j+1},
#                       e.g. "0->1,1->2,2->4,0->3,3->4,0->2"
#   EXAMPLE_SEED        seed for drawing the model from the prior (default 9101)
#   DATA_CFG            required to draw the model or to run the oracle;
#                       otherwise optional (gives the default N_OBS)
#   REPETITIONS         context draws from the fixed model (default 200)
#   K_LIST              default 0,1,3,5,10
#   N_MC                oracle parameter draws per DAG (default 2000)
#   NO_ORACLE=1         evaluate the model against the truth only
#                       (--no_oracle); needed above the oracle's variable limit
#   N_OBS               observational rows (default: from DATA_CFG)
#   DESIGNS             comma-separated design names for the figures, e.g.
#                       "do(X3),random,observational" (quoted: the names
#                       contain parentheses); passes --designs
#   DEVICE              default cpu
#   EXTRA_ARGS          further kshot-example arguments (e.g. "--n_y 32")
#   DRY_RUN=1           print the resolved command and exit (plain `bash` works)
#
# Threads: common_env.sh pins OMP/MKL to 1 thread; this script then sets them
# to SLURM_CPUS_PER_TASK (1 outside Slurm), and kshot-example calls
# torch.set_num_threads(OMP_NUM_THREADS).
#SBATCH --job-name=alcfm_kshot_example
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.err
set -euo pipefail
CKPT=${CKPT:?set CKPT}
MODEL_CFG=${MODEL_CFG:?set MODEL_CFG}
OUT_PREFIX=${OUT_PREFIX:?set OUT_PREFIX}
TREATMENT=${TREATMENT:?set TREATMENT (0-based treatment node index)}
OUTCOME=${OUTCOME:?set OUTCOME (0-based outcome node index)}
DATA_CFG=${DATA_CFG:-}
EXAMPLE_FILE=${EXAMPLE_FILE:-}
FIXED_GRAPH_EDGES=${FIXED_GRAPH_EDGES:-}
EXAMPLE_SEED=${EXAMPLE_SEED:-9101}
REPETITIONS=${REPETITIONS:-200}
K_LIST=${K_LIST:-0,1,3,5,10}
N_MC=${N_MC:-2000}
NO_ORACLE=${NO_ORACLE:-0}
N_OBS=${N_OBS:-}
DESIGNS=${DESIGNS:-}
DEVICE=${DEVICE:-cpu}
EXTRA_ARGS=${EXTRA_ARGS:-}
DRY_RUN=${DRY_RUN:-0}

read -ra EXTRA <<< "$EXTRA_ARGS"
DATA=()
if [ -n "$DATA_CFG" ]; then DATA=(--data_config "$DATA_CFG"); fi
MODEL_SRC=()
if [ -n "$EXAMPLE_FILE" ]; then
  MODEL_SRC=(--example_file "$EXAMPLE_FILE")
elif [ -n "$FIXED_GRAPH_EDGES" ]; then
  MODEL_SRC=(--fixed_graph_edges "$FIXED_GRAPH_EDGES" --example_seed "$EXAMPLE_SEED")
fi
ORACLE=()
if [ "$NO_ORACLE" = 1 ]; then ORACLE=(--no_oracle); fi
NOBS=()
if [ -n "$N_OBS" ]; then NOBS=(--n_obs "$N_OBS"); fi
DESIGN_ARG=()
if [ -n "$DESIGNS" ]; then DESIGN_ARG=(--designs "$DESIGNS"); fi

CMD=(python -u -m CITNP.evaluation kshot-example
  --ckpt "$CKPT" --model_config "$MODEL_CFG"
  --treatment "$TREATMENT" --outcome "$OUTCOME"
  --repetitions "$REPETITIONS" --k_list "$K_LIST" --n_mc "$N_MC"
  --device "$DEVICE"
  ${DATA[@]+"${DATA[@]}"}
  ${MODEL_SRC[@]+"${MODEL_SRC[@]}"}
  ${ORACLE[@]+"${ORACLE[@]}"}
  ${NOBS[@]+"${NOBS[@]}"}
  ${DESIGN_ARG[@]+"${DESIGN_ARG[@]}"}
  ${EXTRA[@]+"${EXTRA[@]}"}
  --out "$OUT_PREFIX.csv")

if [ "$DRY_RUN" = 1 ]; then
  echo "# DRY RUN: kshot-example for X$((OUTCOME + 1)) | do(X$((TREATMENT + 1)))"
  # quote the tokens that contain '>' or '(' so the printed line is runnable
  shown=("${CMD[@]}")
  for i in "${!shown[@]}"; do
    case "${shown[$i]}" in
      *">"* | *"("*) shown[$i]="\"${shown[$i]}\"" ;;
    esac
  done
  echo "${shown[@]}"
  exit 0
fi

source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} \
  MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$(dirname "$OUT_PREFIX")"
echo "=== kshot-example: X$((OUTCOME + 1)) | do(X$((TREATMENT + 1))), $REPETITIONS repetitions"
start=$(date +%s)
"${CMD[@]}"
echo "=== wall time $(($(date +%s) - start)) s"
