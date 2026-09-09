#!/bin/bash
#SBATCH --job-name=phase3
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/train_phase3-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/train_phase3-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${PHASE3_ENV:?Set a modern Python environment with requirements/uptraining.txt installed}"
: "${PHASE3_DATA:?Set the directory containing the token manifest and streams}"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname)"
cd "$HOME/data/VirtualSink"
source "$PHASE3_ENV/bin/activate"
(while sleep 30; do echo "$(date -Is) Still running"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
# The currently advertised K80 partition will fail this prerequisite. Configure
# a BF16-capable allocation and pass a successful GPU test before submitting.
python -u -m virtualsink.uptraining.train \
  --data-dir "$PHASE3_DATA" --output-dir "runs/phase3-$SLURM_JOB_ID" \
  --mode attention --auxiliary anti_collapse --gamma 0.01 \
  --seq-len 2048 --batch-size 1 --accumulation 32 --steps 15000
