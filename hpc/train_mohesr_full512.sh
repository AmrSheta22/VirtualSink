#!/bin/bash
#SBATCH --job-name=smol512_replica
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
: "${MOHESR_EXPERIMENT:?Isolated root required}"
source /nfs/slurm/alex005/VirtualSink/venv-modern/bin/activate
export PYTHONPATH="$MOHESR_EXPERIMENT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
index="${SLURM_ARRAY_TASK_ID:?Array required}"
seed=$((8 + index / 2))
variant=standard
if (( index % 2 )); then variant=query_sink; fi
echo "$(date -Is) START job=$SLURM_JOB_ID seed=$seed variant=$variant host=$(hostname)"
trainer_pid=
trap 'test -z "$trainer_pid" || kill -USR1 "$trainer_pid" 2>/dev/null || true' USR1
python -u -m virtualsink.pretraining.train \
  --data-dir "$MOHESR_EXPERIMENT/data" \
  --output-dir "$MOHESR_EXPERIMENT/seeds/seed$seed/runs/$variant" \
  --variant "$variant" --model-seed "$seed" --steps 30 \
  --microbatch "$(cat "$MOHESR_EXPERIMENT/selected-microbatch.txt")" --chunk-size 32 --sequence-length 512 --window-size 0 \
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
  if [ "${MOHESR_CHAIN:-0}" -eq 1 ]; then exit 75; fi
  echo "$(date -Is) CHECKPOINTED; requeue job=$SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "$(date -Is) FINISHED exit=$code"
exit "$code"
