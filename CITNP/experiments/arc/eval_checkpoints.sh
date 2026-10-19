#!/bin/bash
# Oracle evaluation of every step_*.pt (and best/last) in one run directory.
#   sbatch CITNP/experiments/arc/eval_checkpoints.sh <run_dir> <model_config.yaml>
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=stat-hd-causal
#SBATCH --job-name=alcfm_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

RUN_DIR=$1
MODEL_CFG=$2
REPO_DIR=/data/stat-hd-causal/spet5873/AL_CFM
CONDA_ENV=/data/stat-hd-causal/spet5873/envs/citnp

cd "$REPO_DIR"
module purge; module load Anaconda3; source activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
mkdir -p "$RUN_DIR/eval"

for CKPT in $(ls "$RUN_DIR"/step_*.pt | sort -V) "$RUN_DIR"/best_model.pt "$RUN_DIR"/last_model.pt; do
    [ -f "$CKPT" ] || continue
    TAG=$(basename "$CKPT" .pt)
    echo "=== $TAG"
    "$CONDA_ENV"/bin/python -m CITNP.evaluation scm-suite \
        --ckpt "$CKPT" --model_config "$MODEL_CFG" \
        --data_config CITNP/experiments/configs/oracle_d3.yaml \
        --n_seeds 30 --device cuda \
        --out "$RUN_DIR/eval/${TAG}.csv" > "$RUN_DIR/eval/${TAG}.log"
done

# One line per checkpoint: mean model-minus-oracle gap overall / direct / indirect / obs-only
"$CONDA_ENV"/bin/python - "$RUN_DIR/eval" <<'PY'
import sys, glob, os, re, csv
rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*.csv")):
    tag = os.path.basename(f)[:-4]
    d = list(csv.DictReader(open(f)))
    g = lambda pred: [float(r["model_minus_oracle"]) for r in d if pred(r["design"])]
    m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    step = int(re.findall(r"\d+", tag)[0]) if tag.startswith("step_") else 10**9
    rows.append((step, tag,
                 m(g(lambda s: True)),
                 m(g(lambda s: "do(X0)" in s)),
                 m(g(lambda s: "do(X1)" in s or "do(X2)" in s)),
                 m(g(lambda s: s == "obs_only"))))
print(f"{'checkpoint':14s} {'all':>7s} {'direct':>7s} {'indirect':>8s} {'obs':>7s}")
for _, tag, a, dr, ind, ob in sorted(rows):
    print(f"{tag:14s} {a:+7.3f} {dr:+7.3f} {ind:+8.3f} {ob:+7.3f}")
PY
