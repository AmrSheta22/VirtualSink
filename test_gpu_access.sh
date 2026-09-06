#!/bin/bash
#SBATCH --job-name=compare-long-retrieval
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/test_gpu_access-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/test_gpu_access-%j.err

set -euo pipefail
source "$HOME/data/VirtualSink/hpc_runtime.sh"
status 'RUNNING CUDA diagnostic'
python -u test.py
