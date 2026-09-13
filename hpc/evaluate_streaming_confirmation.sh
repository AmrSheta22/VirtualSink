#!/bin/bash
#SBATCH --job-name=stream_confirm
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR:?}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
index="${SLURM_ARRAY_TASK_ID:?}"
printf -v unit 'article%02d' "$index"
echo "$(date -Is) START job=$SLURM_JOB_ID model=$1 article=$index"
python -u -m virtualsink.uptraining.evaluate_streaming --model "$1" --confirmation-index "$index" --output "$HOME/data/streaming-confirmation-20260913/results/$1/$unit"
