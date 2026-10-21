#!/bin/bash
#SBATCH --job-name=smol512_3seeds
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
export MOHESR_CHAIN=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
runner_pid=
stop_requested=0
trap 'stop_requested=1; test -z "$runner_pid" || kill -USR1 "$runner_pid" 2>/dev/null || true' USR1
indices="0 1 2 3 4 5"
if [ -s "$MOHESR_EXPERIMENT/arm-indices.txt" ]; then
  indices=$(cat "$MOHESR_EXPERIMENT/arm-indices.txt")
fi
for index in $indices; do
  export SLURM_ARRAY_TASK_ID="$index"
  bash "$MOHESR_EXPERIMENT/release/hpc/train_mohesr_full512.sh" &
  runner_pid=$!
  set +e
  while true; do
    wait "$runner_pid"
    code=$?
    if kill -0 "$runner_pid" 2>/dev/null; then continue; fi
    break
  done
  set -e
  runner_pid=
  if [ "$code" -eq 75 ] || [ "$stop_requested" -eq 1 ]; then
    echo "$(date -Is) REQUEUE after checkpoint job=$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [ "$code" -ne 0 ]; then exit "$code"; fi
  if (( index % 2 )); then
    seed=$((8 + index / 2))
    python -m virtualsink.pretraining.compare --root "$MOHESR_EXPERIMENT/seeds/seed$seed"
  fi
done
if [ -f "$MOHESR_EXPERIMENT/seeds/seed8/comparison/comparison.json" ] && \
   [ -f "$MOHESR_EXPERIMENT/seeds/seed9/comparison/comparison.json" ] && \
   [ -f "$MOHESR_EXPERIMENT/seeds/seed10/comparison/comparison.json" ]; then
  python "$MOHESR_EXPERIMENT/release/hpc/summarize_mohesr_full512.py"
else
  echo "$(date -Is) Assigned MOHESR arms complete; other pairs run on Bibalex."
fi
