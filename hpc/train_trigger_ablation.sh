#!/bin/bash
#SBATCH --job-name=trigger_ablation
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/train_trigger_ablation-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/train_trigger_ablation-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch on the HPC}"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname)"
cd "$HOME/data/VirtualSink"
source venv-k80/bin/activate
echo "Python: $(command -v python)"
(while sleep 30; do echo "$(date -Is) Still running"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -m pytest -q test_trigger_experiment.py
python -u run_ablation.py --device cpu --threads "$SLURM_CPUS_PER_TASK" \
    --output-dir "runs/trigger-${SLURM_JOB_ID}"
