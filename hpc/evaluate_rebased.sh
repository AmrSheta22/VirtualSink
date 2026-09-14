#!/bin/bash
#SBATCH --job-name=smol_rebase
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR:?}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u -m pytest -q tests/test_rebased_streaming.py tests/test_uptrained_streaming.py
python -u -m virtualsink.uptraining.evaluate_rebased
