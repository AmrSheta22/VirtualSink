#!/bin/bash
#SBATCH --job-name=qwen15_seed
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
set -euo pipefail
cd "${QWEN_SEEDS_ROOT:?}/release"
source hpc/hpc_runtime.sh
export PYTHONPATH="$QWEN_SEEDS_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
index="${SLURM_ARRAY_TASK_ID:?}"
seed=$((1729 + index / 2))
variant=standard
if ((index % 2)); then variant=query_sink; fi
run="$QWEN_SEEDS_ROOT/seeds/seed$seed/runs/$variant"
if [ -e "$run/summary.json" ]; then status "ALREADY_COMPLETE"; exit 0; fi
status "RUNNING model=Qwen2.5-1.5B seed=$seed variant=$variant"
resume=()
if [ -e "$run/last.pt" ]; then resume=(--resume "$run/last.pt"); fi
trainer_pid=
trap 'test -z "$trainer_pid" || kill -USR1 "$trainer_pid" 2>/dev/null || true' USR1
python -u -m virtualsink.uptraining.train \
  --backend native --precision fp32 --data-dir "$QWEN_SEEDS_ROOT/data" \
  --output-dir "$run" --mode attention --variant "$variant" \
  --auxiliary anti_collapse --gamma .01 --seed "$seed" \
  --steps 1000 --seq-len 1024 --window-size 512 --batch-size 1 --accumulation 8 \
  --chunk-size 32 --eval-every 25 --eval-batches 32 --save-every 25 \
  --save-seconds 1800 --milestone-every 1000 --max-hours 47.5 "${resume[@]}" &
trainer_pid=$!
set +e
while true; do
  wait "$trainer_pid"
  code=$?
  if kill -0 "$trainer_pid" 2>/dev/null; then continue; fi
  break
done
set -e
if [ "$code" -ne 0 ]; then exit "$code"; fi
if [ ! -e "$run/summary.json" ]; then
  test -e "$run/last.pt"
  status "CHECKPOINTED; requeue job=$SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
