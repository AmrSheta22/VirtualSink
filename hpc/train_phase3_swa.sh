#!/bin/bash
#SBATCH --job-name=phase3_swa
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@300
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SWA_ROOT:?Set the experiment root}"
release="${SLURM_SUBMIT_DIR:?Submit from release}"
base="$HOME/data/phase3-k80-20260909"
slug="${1:?smollm135 or qwen05}"
variant="${2:?query_sink or standard}"
case "$slug" in smollm135) mode=full ;; qwen05) mode=attention ;; *) exit 2 ;; esac
case "$variant" in query_sink|standard) ;; *) exit 2 ;; esac
source "$base/environment/bin/activate"
export PYTHONPATH="$release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START job=$SLURM_JOB_ID host=$(hostname) model=$slug variant=$variant"
nvidia-smi
python -u -c 'import torch; torch.cuda.init(); x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize(); print("GPU forward/backward OK",flush=True)'
(while sleep 30; do echo "$(date -Is) Job active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
python -u -m pytest -q "$release/tests/test_swa_uptraining.py"
output="$SWA_ROOT/runs/$slug-$variant"
mkdir -p "$SWA_ROOT/runs"
python -u -m virtualsink.uptraining.train \
  --backend native --precision fp32 --data-dir "$base/data/$slug" \
  --output-dir "$output" --mode "$mode" --variant "$variant" \
  --auxiliary anti_collapse --gamma 0.01 --steps 1000 --seq-len 1024 --window-size 512 \
  --batch-size 1 --accumulation 8 --chunk-size 64 \
  --eval-every 25 --eval-batches 32 --save-every 10 --save-seconds 600 \
  --milestone-every 250 --max-hours 23.5
