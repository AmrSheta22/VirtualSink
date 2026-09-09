#!/bin/bash
#SBATCH --job-name=sink_eval
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
release="${SLURM_SUBMIT_DIR:?Submit from release}"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START job=$SLURM_JOB_ID host=$(hostname)"
(while sleep 30; do echo "$(date -Is) Evaluation active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
python -u -m virtualsink.uptraining.evaluate_sinks "$@"
