#!/bin/bash
#SBATCH --job-name=smol_scratch
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
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SMOL_PRETRAIN_ROOT:?Set isolated experiment root}"
: "${SMOL_PRETRAIN_STEPS:?Set matched step budget}"
variant="${1:?Pass standard or query_sink}"
case "$variant" in standard|query_sink) ;; *) exit 2 ;; esac
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$SMOL_PRETRAIN_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
echo "$(date -Is) START job=$SLURM_JOB_ID host=$(hostname) variant=$variant restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi
python -u -c 'import torch; torch.cuda.init(); x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize(); print("GPU FORWARD/BACKWARD PASSED",torch.__version__,torch.cuda.get_device_name(),flush=True)'
(while sleep 30; do echo "$(date -Is) Training shell active"; done) &
heartbeat=$!
trainer_pid=
trap 'test -z "$trainer_pid" || kill -USR1 "$trainer_pid" 2>/dev/null || true' USR1
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"' EXIT
python -u -m virtualsink.pretraining.train --data-dir "$SMOL_PRETRAIN_ROOT/data" \
  --output-dir "$SMOL_PRETRAIN_ROOT/runs/$variant" --variant "$variant" \
  --steps "$SMOL_PRETRAIN_STEPS" --microbatch 1 --chunk-size 32 \
  --sequence-length "${SMOL_SEQUENCE_LENGTH:-8192}" --window-size "${SMOL_WINDOW_SIZE:-512}" --sink-weight-std 0.02 --sink-bias 4.0 \
  --eval-every 1 --eval-blocks "${SMOL_EVAL_BLOCKS:-4}" --save-seconds 600 --max-hours 23 &
trainer_pid=$!
# A trapped warning can interrupt wait before Python finishes checkpointing.
set +e
while true; do
  wait "$trainer_pid"
  code=$?
  if kill -0 "$trainer_pid" 2>/dev/null; then continue; fi
  break
done
set -e
if [ "$code" -eq 75 ]; then
  echo "$(date -Is) Checkpointed pause; requeueing job=$SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
