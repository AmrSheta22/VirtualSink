#!/bin/bash
#SBATCH --job-name=smol512_replica
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
: "${SPLIT_EXPERIMENT:?Set isolated Bibalex experiment root}"
cd "$SPLIT_EXPERIMENT/release"
source hpc/hpc_runtime.sh
export PYTHONPATH="$SPLIT_EXPERIMENT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
index="${SLURM_ARRAY_TASK_ID:?Array required}"
seed=$(( ${SPLIT_SEED_START:-9} + index / 2 ))
variant=standard
if (( index % 2 )); then variant=query_sink; fi
run="$SPLIT_EXPERIMENT/seeds/seed$seed/runs/$variant"
status "RUNNING seed=$seed variant=$variant restart=${SLURM_RESTART_COUNT:-0}"
if [ ! -e "$run/last.pt" ] && [ ! -e "$run/summary.json" ]; then
  python -m pytest -q tests/test_native_uptraining.py tests/test_pretraining.py tests/test_streaming.py
  python -u hpc/smoke_bibalex_replication.py "$variant"
fi
trainer_pid=
trap 'test -z "$trainer_pid" || kill -USR1 "$trainer_pid" 2>/dev/null || true' USR1
python -u -m virtualsink.pretraining.train \
  --data-dir "$SPLIT_EXPERIMENT/data" --output-dir "$run" \
  --variant "$variant" --model-seed "$seed" --steps 30 --microbatch 16 \
  --chunk-size 32 --sequence-length 512 --window-size 0 \
  --sink-weight-std .02 --sink-bias 4 --eval-every 1 --eval-blocks 64 \
  --save-seconds 600 --max-hours 23 &
trainer_pid=$!
set +e
while true; do
  wait "$trainer_pid"
  code=$?
  if kill -0 "$trainer_pid" 2>/dev/null; then continue; fi
  break
done
set -e
if [ "$code" -eq 75 ]; then
  status "CHECKPOINTED; requeue job=$SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
