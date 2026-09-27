#!/bin/bash
#SBATCH --job-name=swa_head_ablation
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
root="$HOME/data/smollm-swa8192-20260911"
export PYTHONPATH="$root/release/src"
export HF_HOME="$root/hf-cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u "$root/head-interventions/head_intervention.py" --root "$root" --output "$root/head-interventions/results"
