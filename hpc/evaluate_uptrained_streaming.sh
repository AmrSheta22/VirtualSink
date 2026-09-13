#!/bin/bash
#SBATCH --job-name=uptrained_stream
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR:?}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START $SLURM_JOB_ID $(hostname)"
python -u -m pytest -q tests/test_uptrained_streaming.py
python -u -m virtualsink.uptraining.evaluate_streaming --model "$1" --output "$HOME/data/uptrained-streaming-20260913/results/$1"
