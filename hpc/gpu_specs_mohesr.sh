#!/bin/bash
#SBATCH --job-name=virtualsink-gpu-specs
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/slurm/alex005/VirtualSink/hpc/%x-%j.out
#SBATCH --error=/nfs/slurm/alex005/VirtualSink/hpc/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
source /nfs/slurm/alex005/VirtualSink/venv-modern/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /nfs/slurm/alex005/VirtualSink/releases/20261009
python -u /nfs/slurm/alex005/VirtualSink/hpc/gpu_specs_mohesr.py
python -m pytest -q
