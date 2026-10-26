#!/bin/bash
# Run the pytest suite (and pre-commit on the last commit's files) on a CPU
# node. Set the options in the environment, e.g.
#
#   mkdir -p /gpfs3/well/cebam/users/$USER/alcfm/logs
#   TESTS="testing/test_a.py testing/test_b.py" MARK="not slow" \
#       sbatch CITNP/experiments/bmrc/run_tests.sh
#
#   TESTS  pytest targets                   (default: testing)
#   MARK   pytest -m expression             (default: not slow)
#   EXTRA  further pytest arguments         (default: none)
#
# Check the CPU partition and account names with `sinfo -s` and
# `sacctmgr show assoc user=$USER`; add `#SBATCH --account=...` if required.
# Add mail options on the sbatch command line (--mail-type, --mail-user).
#SBATCH --job-name=alcfm_tests
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.out
#SBATCH --error=/gpfs3/well/cebam/users/%u/alcfm/logs/%x_%j.err
set -euo pipefail
TESTS=${TESTS:-testing}
MARK=${MARK:-not slow}
EXTRA=${EXTRA:-}
source "${COMMON_ENV:-${SLURM_SUBMIT_DIR:-.}/CITNP/experiments/bmrc/common_env.sh}"

echo "=== pytest $TESTS -m \"$MARK\" -q $EXTRA"
status=0
# shellcheck disable=SC2086  # TESTS and EXTRA are word-split on purpose
python -m pytest $TESTS -m "$MARK" -q $EXTRA || status=$?

if command -v pre-commit >/dev/null 2>&1; then
  echo "=== pre-commit on the files of the last commit"
  # shellcheck disable=SC2046
  pre-commit run --files $(git diff --name-only --diff-filter=d HEAD~1) \
    || status=$?
fi
echo "=== exit status $status"
exit "$status"
