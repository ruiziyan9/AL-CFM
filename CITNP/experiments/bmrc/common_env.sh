# Sourced helper (not an sbatch script): environment shared by the BMRC
# launchers. Every path can be overridden from the calling environment:
#
#   W        base directory               (default /gpfs3/well/cebam/users/$USER)
#   REPO     checkout of this repository  (default $W/AL-CFM)
#   OUT      logs, banks, results         (default $W/alcfm)
#   CONDA_ENV, MODULE                     (defaults mace_tnp, Miniforge3/24.11.3-0)
#
# Usage, from a batch script or an interactive `srun --pty bash` shell:
#   source CITNP/experiments/bmrc/common_env.sh
case $- in *u*) _had_nounset=1 ;; *) _had_nounset=0 ;; esac
set +u
USER=${USER:-$(id -un)}
W=${W:-/gpfs3/well/cebam/users/$USER}
REPO=${REPO:-$W/AL-CFM}
OUT=${OUT:-$W/alcfm}
CONDA_ENV=${CONDA_ENV:-mace_tnp}
MODULE=${MODULE:-Miniforge3/24.11.3-0}
export W REPO OUT CONDA_ENV MODULE
mkdir -p "$OUT/logs"
cd "$REPO"
module purge
module load "$MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
ulimit -n "$(ulimit -Hn)" || true
if [ "$_had_nounset" = 1 ]; then set -u; fi
unset _had_nounset
echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname), $(date)"
echo "=== Python: $(which python)"
echo "=== Git: $(git rev-parse --short HEAD)"
git status --short
