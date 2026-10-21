---
name: bibalex-hpc
description: >-
  Workflows, procedures, and runbooks for submitting and monitoring Slurm jobs on
  the Bibalex HPC cluster (login02.c2.hpc.bibalex.org). Use this skill when
  running training, CUDA diagnostics, or batch jobs on Bibalex HPC.
---

# Bibalex HPC Runbook & Workflow

This skill outlines how to interact with, submit jobs to, and monitor executions on the Bibalex HPC cluster for the `VirtualSink` project.

---

## 1. Cluster Connection & Credentials

- **Login Host**: `login02.c2.hpc.bibalex.org`
- **SSH User**: `alex116u1`
- **Private Key**: `server_access/private_key` (ignored in Git)
- **Known Hosts**: `server_access/known_hosts` (ignored in Git)
- **Remote Workdir**: `~/data/VirtualSink`
- **Remote Python venv**: `~/data/VirtualSink/venv`
- **Slurm Account**: `g.alex116`
- **Excluded Nodes**: `comp013` (known issue node)

> [!IMPORTANT]
> Never place the SSH private key in version-controlled scripts or upload it to the HPC. Never execute long-running training jobs directly on the login node or local machine.

---

## 2. Writing Slurm Batch Scripts

All batch scripts must source `hpc/hpc_runtime.sh`, which enforces execution inside Slurm, activates `~/data/VirtualSink/venv`, logs status timestamps, and manages heartbeats.

### Standard Template (`hpc/train_job.sh` or `hpc/test_shell.sh`)
```bash
#!/bin/bash
#SBATCH --job-name=virtualsink-job
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/%x-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/%x-%j.err

set -euo pipefail
source "$HOME/data/VirtualSink/hpc/hpc_runtime.sh"

status 'RUNNING experiment'
python -u train_script.py --args
```

### Directives Checklist
- All `#SBATCH` directives must precede executable commands.
- Use `%j` for Slurm Job ID and `%x` for Job Name.
- Do not use shell variables (such as `$HOME`) inside `#SBATCH` directives.
- Always save batch shell scripts with **LF** (Unix) line endings.

---

## 3. Remote Code Sync & Submission

### A. Submitting via PowerShell (`submit_hpc.ps1`)
```powershell
.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh
```

### B. Submitting via Bash / SSH
```bash
ssh -i server_access/private_key -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    alex116u1@login02.c2.hpc.bibalex.org \
    "cd ~/data/VirtualSink && git pull && cd hpc && sbatch test_gpu_access.sh"
```

---

## 4. Job Monitoring & Verification Protocol

### Required Execution Confirmation
Every submission must be followed by inspecting the job's `.out` file **at least 20 seconds after submission**.
A job ID alone is insufficient to confirm execution.

```bash
# Check queue status
ssh -i server_access/private_key alex116u1@login02.c2.hpc.bibalex.org "squeue -u alex116u1"

# Check accounting
ssh -i server_access/private_key alex116u1@login02.c2.hpc.bibalex.org "sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed"

# Read logs
ssh -i server_access/private_key alex116u1@login02.c2.hpc.bibalex.org "tail -n 80 ~/data/VirtualSink/hpc/*-<JOB_ID>.out"
ssh -i server_access/private_key alex116u1@login02.c2.hpc.bibalex.org "tail -n 80 ~/data/VirtualSink/hpc/*-<JOB_ID>.err"
```

### Status Markers emitted by `hpc_runtime.sh`:
- `STARTED`: Batch shell initialized on compute node.
- `ENV_READY`: Virtual environment activated successfully.
- `RUNNING`: Main command launched.
- `HEARTBEAT`: Emitted periodically every 10s.
- `COMPLETED exit_code=0`: Normal termination.
- `FAILED exit_code=N`: Execution failed; inspect error log.
