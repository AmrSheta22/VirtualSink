#!/bin/bash
#SBATCH --job-name=qwen_phys_sink
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR:?}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START $SLURM_JOB_ID $(hostname)"
(while sleep 30; do echo "$(date -Is) Probe active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
python -u -m virtualsink.uptraining.physical_sink_probe
