---
name: mohesr-hpc
description: >-
  Workflows, procedures, and runbooks for accessing, submitting, and monitoring
  jobs on the MOHESR HPC cluster (hpclogin.mohesr.gov.eg). Use this skill when
  running training, CUDA/GPU diagnostics, or Slurm batch jobs on MOHESR HPC.
---

# MOHESR HPC Runbook & Workflow

This skill outlines how to interact with, submit jobs to, and monitor executions on the MOHESR HPC cluster (`hpclogin.mohesr.gov.eg`).

---

## 1. Connection & Credentials (2FA)

Accessing MOHESR HPC requires **Two-Factor Authentication (SSH Key + Password)**:

| Setting | Value / Environment Variable |
| --- | --- |
| **SSH Address** | `alex005@hpclogin.mohesr.gov.eg` |
| **Login Node** | `login2.slurm.mohesr.gov.eg` |
| **Private Key** | Path in `$MOHESR_HPC_KEY` (e.g. `/home/sheta/private2`) |
| **Password** | Secret stored in `$MOHESR_HPC_PASSWORD` |

| **Initial Login Directory** | `/nfs/slurm/alex005` |
| **Remote Repository** | `/nfs/slurm/alex005/VirtualSink` |

### Network & Security Precautions
- **VPN / Cloudflare WARP**: Ensure Cloudflare WARP (`warp-cli connect`) or a local VPN is active when connecting.
- **Fail2Ban / Rate Limiting**: Entering an incorrect password 3 times triggers a temporary connection ban on port 22 (`Connection refused`) lasting 5 to 10 minutes.

---

## 2. Cluster Architecture & Slurm Partitions

| Partition | Node | Hardware / Accelerator | GRES Specification |
| --- | --- | --- | --- |
| `gpu5*` (Default) | `gpu5srv` | 8x NVIDIA A100 80GB PCIe (32 MIG slices) | `--gres=gpu:1` (`a100_1g.20gb`) |
| `main` | `gpu[1,3]` | NVIDIA A100 GPUs (16 MIG slices) | `--gres=gpu:1` |

Each `--gres=gpu:1` allocation provides:
- Accelerator: **NVIDIA A100 80GB PCIe MIG 1g.20gb**
- Dedicated VRAM: **~19.50 GB**
- Compute Capability: **8.0**
- Driver Version: **570.211.01** (supports CUDA 12.x)

---

## 3. Slurm Submission Filter Policies (Important!)

> [!WARNING]
> The MOHESR Slurm cluster enforces a `job_submit` plugin that parses script text and **blocks jobs containing specific commands**:
> - **DO NOT include `nvidia-smi`** in any batch script. It will be rejected immediately with:
>   `sbatch: error: ERROR: Blocked command found in job script. Command: nvidia-smi`
>
> **Recommended GPU Diagnostic Workarounds**:
> 1. Read hardware details directly from `/proc/driver/nvidia/version` and `/proc/driver/nvidia/gpus/*/information`.
> 2. Query CUDA Driver API directly from Python via `ctypes` (`libcuda.so.1`) using `cuInit`, `cuDeviceGetCount`, etc.
> 3. Check PyTorch CUDA via `torch.cuda.is_available()`.

---

## 4. Standard Batch Script Template

Save as `/nfs/slurm/alex005/VirtualSink/hpc/job_mohesr.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=mohesr-job
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/slurm/alex005/%x-%j.out
#SBATCH --error=/nfs/slurm/alex005/%x-%j.err

set -euo pipefail

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date -Is)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================================="

cd /nfs/slurm/alex005/VirtualSink

# Run workload (avoid blocked keywords like nvidia-smi)
python3 -u run_ablation.py
```

---

## 5. Automated Submission Helper (`hpc/submit_mohesr.py`)

Because the server requires both an SSH key and interactive password authentication, use `python3 hpc/submit_mohesr.py` (which leverages `pexpect` and environment variables) to submit and track jobs automatically without typing passwords repeatedly.

```bash
# Submit a script
python3 hpc/submit_mohesr.py --script hpc/cuda_check.sh

# Submit and wait for logs
python3 hpc/submit_mohesr.py --script hpc/cuda_check.sh --watch
```
