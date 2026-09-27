#!/bin/bash
#SBATCH --job-name=smol_compare
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SMOL_PRETRAIN_ROOT:?Set isolated experiment root}"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$SMOL_PRETRAIN_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START comparison job=$SLURM_JOB_ID host=$(hostname)"
(while sleep 30; do echo "$(date -Is) Comparison shell active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
python -u -m virtualsink.pretraining.compare --root "$SMOL_PRETRAIN_ROOT"
