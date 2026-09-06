#!/bin/bash
#SBATCH --job-name=virtualsink-smoke
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/test_shell-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/test_shell-%j.err

set -euo pipefail
source "$HOME/data/VirtualSink/hpc_runtime.sh"
status 'RUNNING smoke test'
echo "Hello first"
