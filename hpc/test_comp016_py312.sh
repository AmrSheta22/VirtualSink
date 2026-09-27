#!/bin/bash
#SBATCH --job-name=test_py312_cuda
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=comp016
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/test_py312_cuda-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/test_py312_cuda-%j.err

set -u
: "${SLURM_JOB_ID:?Submit this script with sbatch on the HPC}"
echo "$(date -Is) STARTED job=$SLURM_JOB_ID host=$(hostname)"
(while sleep 10; do echo "$(date -Is) Still running"; done) &
heartbeat=$!
trap 'code=$?; kill "$heartbeat" 2>/dev/null || true; echo "$(date -Is) FINISHED exit=$code"; exit "$code"' EXIT

echo "========================================="
echo "1. MODULE LOADING: Python 3.12 + CUDA 12.4"
echo "========================================="
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.4.0
module list

echo "Python: $(which python3)"
python3 --version
echo "NVCC:"
nvcc -V || true

echo ""
echo "========================================="
echo "2. NVIDIA-SMI STATUS"
echo "========================================="
timeout 20s nvidia-smi || echo "nvidia-smi failed or timed out"

echo ""
echo "========================================="
echo "3. PYTORCH TEST (Python 3.12 + CUDA 12.4)"
echo "========================================="
python3 -u - <<'PY'
import sys
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("Torch CUDA runtime:", torch.version.cuda)
    print("CUDA is_available:", torch.cuda.is_available())
    print("CUDA device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        print("Tensor test on CUDA successful:", t)
    else:
        print("torch.cuda.is_available() is False. Attempting torch.cuda.init()...")
        torch.cuda.init()
except Exception as e:
    print("PyTorch error:", repr(e))
    import traceback
    traceback.print_exc()
PY

echo ""
echo "========================================="
echo "4. LOW-LEVEL DRIVER & NVML DIAGNOSTICS"
echo "========================================="
python3 -u - <<'PY'
import ctypes

for libname in ['libcuda.so.1', 'libnvidia-ml.so.1']:
    try:
        lib = ctypes.CDLL(libname)
        print(f"Loaded {libname}: OK")
    except Exception as e:
        print(f"Failed loading {libname}: {e}")

try:
    libcuda = ctypes.CDLL('libcuda.so.1')
    v = ctypes.c_int()
    libcuda.cuDriverGetVersion(ctypes.byref(v))
    print(f"Driver API version: {v.value}")
    init_res = libcuda.cuInit(0)
    print(f"cuInit(0) return code: {init_res} (0 = SUCCESS, 100 = NO_DEVICE/UNKNOWN)")
except Exception as e:
    print("cuInit call error:", e)

try:
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    nvml_init_res = nvml.nvmlInit()
    print(f"nvmlInit() return code: {nvml_init_res} (0 = SUCCESS)")
    if nvml_init_res == 0:
        count = ctypes.c_uint()
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
        print(f"nvmlDeviceGetCount: {count.value}")
        handle = ctypes.c_void_p()
        h_res = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        print(f"nvmlDeviceGetHandleByIndex(0): {h_res}")
except Exception as e:
    print("nvml call error:", e)
PY

echo ""
echo "========================================="
echo "5. TESTING WITH CUDA 11.7.0"
echo "========================================="
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0
module list
python3 -u - <<'PY'
import ctypes
try:
    libcuda = ctypes.CDLL('libcuda.so.1')
    print("CUDA 11.7 cuInit(0) return code:", libcuda.cuInit(0))
except Exception as e:
    print("Error:", e)
PY

echo ""
echo "========================================="
echo "6. HARDWARE & KERNEL BUS DIAGNOSTICS"
echo "========================================="
echo "Kernel driver version:"
cat /proc/driver/nvidia/version 2>/dev/null || echo "Unable to read driver version"

echo "Device nodes in /dev:"
ls -la /dev/nvidia* 2>/dev/null || true

echo "GPU information in /proc/driver/nvidia/gpus/:"
for f in /proc/driver/nvidia/gpus/*/information; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        cat "$f"
    fi
done

echo "PCI bus status for NVIDIA devices:"
lspci -nnk -d 10de: 2>/dev/null || true

echo "Recent kernel messages (dmesg):"
dmesg -T 2>/dev/null | grep -Ei 'NVRM|Xid|nvidia' | tail -n 30 || echo "dmesg restricted or empty"

echo "$(date -Is) COMPLETED ALL DIAGNOSTICS"
