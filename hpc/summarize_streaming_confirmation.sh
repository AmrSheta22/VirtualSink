#!/bin/bash
#SBATCH --job-name=stream_report
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR:?}/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m virtualsink.uptraining.summarize_streaming_confirmation --root "$HOME/data/streaming-confirmation-20260913"
