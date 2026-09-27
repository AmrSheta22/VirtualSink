#!/bin/bash
#SBATCH --job-name=test_gpu_access
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/test_gpu_access-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/test_gpu_access-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch on the HPC}"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname)"
cd "$HOME/data/VirtualSink"
source venv-k80/bin/activate
echo "Python: $(command -v python)"
(while sleep 10; do echo "$(date -Is) Still running"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT

timeout 20s nvidia-smi || echo "GPU driver check failed"
python -u - <<'PY'
import torch

print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
torch.cuda.init()  # Fail with the actual CUDA error instead of silently using CPU.
print("GPU:", torch.cuda.get_device_name(0))
print("Tensor test:", torch.tensor([1, 2, 3], device="cuda"))
PY
