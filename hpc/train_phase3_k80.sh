#!/bin/bash
#SBATCH --job-name=phase3_k80
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
: "${PHASE3_ROOT:?Set the isolated experiment root}"
slug="${1:?Pass smollm135 or qwen05}"
case "$slug" in
  smollm135) mode=full ;;
  qwen05) mode=attention ;;
  *) echo "Unknown model $slug"; exit 2 ;;
esac
release="$(cd "$(dirname "$0")/.." && pwd)"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname) model=$slug mode=$mode release=$release"
source "$PHASE3_ROOT/environment/bin/activate"
export PYTHONPATH="$release/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$PHASE3_ROOT/hf-cache"
output="$PHASE3_ROOT/runs/$slug"
mkdir -p "$PHASE3_ROOT/runs"
nvidia-smi
python -u -c 'import torch; torch.cuda.init(); x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize(); print("GPU FORWARD/BACKWARD PASSED", torch.__version__, torch.cuda.get_device_name(), flush=True)'
(while sleep 30; do echo "$(date -Is) Training shell active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
resume=()
if [ -f "$output/last.pt" ]; then resume=(--resume "$output/last.pt"); fi
python -u -m virtualsink.uptraining.train \
  --backend native --precision fp32 --data-dir "$PHASE3_ROOT/data/$slug" \
  --output-dir "$output" --mode "$mode" --variant query_sink \
  --auxiliary anti_collapse --gamma 0.01 --steps 1000 --seq-len 256 \
  --batch-size 1 --accumulation 32 --chunk-size 64 \
  --eval-every 25 --eval-batches 32 --save-every 10 --save-seconds 600 \
  --milestone-every 250 --max-hours 23.5 "${resume[@]}"
