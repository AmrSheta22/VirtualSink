#!/bin/bash
#SBATCH --job-name=smol_stream_diag
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$SMOL_PRETRAIN_ROOT/release/src"
export HF_HOME="$HOME/data/smollm-swa8192-20260911/hf-cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
set +e
python -u -m virtualsink.pretraining.diagnose_streaming --root "$1" --output-dir "$2" --window 512 --length 32768 --streams 2
code=$?
set -e
if [ "$code" -eq 75 ]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$code"
