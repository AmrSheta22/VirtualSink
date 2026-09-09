#!/bin/bash
#SBATCH --job-name=phase3_prepare
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${PHASE3_ROOT:?Set the isolated experiment root}"
release="$(cd "$(dirname "$0")/.." && pwd)"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname) release=$release"
(while sleep 30; do echo "$(date -Is) Preparation still running"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT
base="$HOME/data/VirtualSink/venv-k80"
environment="$PHASE3_ROOT/environment"
"$base/bin/python" -m venv "$environment"
# Reuse the verified legacy Torch install without changing its environment.
printf '%s\n' "$base/lib/python3.9/site-packages" > "$environment/lib/python3.9/site-packages/k80-base.pth"
source "$environment/bin/activate"
python -m pip install -r "$release/requirements/k80-uptraining.txt"
python -m pip check
export PYTHONPATH="$release/src"
export HF_HOME="$PHASE3_ROOT/hf-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
python -u -m pytest -q "$release/tests/test_uptraining.py" "$release/tests/test_native_uptraining.py"
python -u -m virtualsink.uptraining.prepare_k80 --output-dir "$PHASE3_ROOT/data" --tokens 8192000
python -m pip freeze > "$PHASE3_ROOT/environment-freeze.txt"
