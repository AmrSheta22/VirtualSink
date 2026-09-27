#!/bin/bash
#SBATCH --job-name=smol_scratch_data
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SMOL_PRETRAIN_ROOT:?Set isolated experiment root}"
: "${SMOL_PRETRAIN_STEPS:?Set matched step budget}"
: "${SMOL_TOKENIZER_REVISION:?Set pinned tokenizer revision}"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$SMOL_PRETRAIN_ROOT/release/src"
export HF_HOME="$SMOL_PRETRAIN_ROOT/hf-cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$(date -Is) START preparation job=$SLURM_JOB_ID host=$(hostname)"
(while sleep 30; do echo "$(date -Is) Preparation shell active"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
python -u -m pytest -q "$SMOL_PRETRAIN_ROOT/release/tests/test_pretraining.py"
python -u -m virtualsink.pretraining.data --output-dir "$SMOL_PRETRAIN_ROOT/data" \
  --steps "$SMOL_PRETRAIN_STEPS" --tokenizer-revision "$SMOL_TOKENIZER_REVISION" \
  --sequence-length "${SMOL_SEQUENCE_LENGTH:-8192}" --allow-stackoverflow-substitution
