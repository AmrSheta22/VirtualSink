# Running VirtualSink experiments on Bibalex HPC

All training, GPU tests, and experiment execution must run on HPC compute nodes through Slurm. Use this Windows machine to edit and transfer files. Use the HPC login node to update the checkout, submit jobs, and inspect results. Never start training with local `python`, local `bash`, or directly inside an SSH login session.

## Connection and paths

| Setting | Value |
| --- | --- |
| SSH account | `alex116u1@login02.c2.hpc.bibalex.org` |
| Local private key | `server_access/private_key` |
| Local host trust file | `server_access/known_hosts` |
| Remote repository | `~/data/VirtualSink` |
| Remote virtual environment | `~/data/VirtualSink/venv` |
| Slurm account | `g.alex116` |

Keep `server_access/` out of Git. The repository `.gitignore` excludes it. Never put the private key in a script or upload it to the HPC.

## Write a batch script

Both batch scripts source `hpc_runtime.sh`, which activates `~/data/VirtualSink/venv`, refuses execution outside Slurm, and prints timestamped status. Copy this template for new jobs. Save shell files as UTF-8 without a BOM with LF line endings.

```bash
#!/bin/bash
#SBATCH --job-name=virtualsink-smoke
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/users/alex116u1/data/VirtualSink/hpc/test_shell-%j.out
#SBATCH --error=/cluster/users/alex116u1/data/VirtualSink/hpc/test_shell-%j.err

set -euo pipefail
source "$HOME/data/VirtualSink/hpc/hpc_runtime.sh"
status 'RUNNING smoke test'
echo "Hello first"

```

Keep every `#SBATCH` directive above the first executable command. `%j` in log names becomes the job ID. Shell variables such as `$HOME` are not expanded inside directives. See the [official sbatch reference](https://slurm.schedmd.com/sbatch.html).

For a training script, copy this template to `train_job.sh`, choose a descriptive job name, and set a suitable time limit. Replace the last `echo` with your actual training command, for example:

```bash
# Example only: train.py must exist and accept these arguments.
mkdir -p "runs/$SLURM_JOB_ID"
python -u train.py --output-dir "runs/$SLURM_JOB_ID"
```

`train.py` is a placeholder, not an existing training entry point verified by this guide. Use the real entry point and its supported options. The `-u` flag makes Python output available promptly in the job log. Store checkpoints and results on the remote shared filesystem, using separate output directories per job.

For the existing CUDA diagnostic, use `python -u hpc/test.py` inside the batch script, or submit `test_gpu_access.sh`. Ensure dependencies are installed in the remote virtual environment before submitting. Request CPUs and memory appropriate to your experiment using site-supported settings; requesting more GPUs alone does not make a Python program distributed. The `comp013` exclusion is inherited from the existing GPU script.

## Make code available remotely

`git pull` fetches committed and pushed changes only. It does not copy local untracked files. Commit and push the intended scripts and source files through your normal Git workflow before submission. Do not commit the key, virtual environment, datasets, or generated checkpoints.

For a one-off test with uncommitted files, copy only the required files from PowerShell at the repository root:

```powershell
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"
& $scp -i server_access/private_key -o IdentitiesOnly=yes `
    -o ControlMaster=no -o ControlPath=none -o ControlPersist=no `
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile=server_access/known_hosts `
    hpc/test_gpu_access.sh hpc/test.py hpc/hpc_runtime.sh `
    'alex116u1@login02.c2.hpc.bibalex.org:data/VirtualSink/hpc/'
if ($LASTEXITCODE -ne 0) { throw 'File transfer failed.' }
```

Save files with LF before copying; SCP preserves their bytes. This command replaces remote files with the same names. Avoid modifying the shared checkout while queued or running jobs rely on it: Slurm retains the submitted batch script, but your Python files are still read from the filesystem. For concurrent experiments, use separate remote checkouts and point each batch script to its own checkout.

## Submit and disconnect

After preparing and syncing a valid `test_shell.sh`, run the existing helper from PowerShell:

```powershell
.\hpc\submit_hpc.ps1
```

It connects to the HPC and executes:

```bash
cd ~/data/VirtualSink && git pull && cd hpc && sbatch test_shell.sh
```

The `&&` chain prevents submission if the update fails. Use `.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh` or `-Script train_job.sh` to select another batch script in `hpc/`. The helper sets absolute log paths beside the remote script, waits 20 seconds after submission returns, and opens a fresh SSH connection to read output, errors, and job state.

To submit another script or inspect jobs, define this PowerShell helper at the repository root:

```powershell
function Invoke-Hpc {
    param([Parameter(Mandatory)][string]$Command)
    & "$env:WINDIR\System32\OpenSSH\ssh.exe" -T `
        -i server_access/private_key -o IdentitiesOnly=yes -o BatchMode=yes `
        -o ControlMaster=no -o ControlPath=none -o ControlPersist=no `
        -o StrictHostKeyChecking=yes -o UserKnownHostsFile=server_access/known_hosts `
        -o ConnectTimeout=20 'alex116u1@login02.c2.hpc.bibalex.org' $Command
    if ($LASTEXITCODE -ne 0) { throw "HPC command failed: $LASTEXITCODE" }
}

.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh
# For a training script you have created and synced:
# .\hpc\submit_hpc.ps1 -Script train_job.sh
```

Single quotes preserve remote `$HOME` and other shell variables from PowerShell expansion. These examples use the host key already recorded during connection setup. If host verification fails, verify the host key with the HPC administrator before changing the trust record.

Successful submission prints `Submitted batch job <ID>`. This confirms acceptance, not successful execution. The SSH command exits and closes its connection after submission, including when the remote command fails. Slurm manages the job independently; no open terminal is required.

## Check progress and results

### Required execution confirmation for users and LLM agents

Every submission must be followed by an actual read of the job's `.out` file, no earlier than **20 seconds after submission**. The submission helper implements the delay and first read attempt. The LLM must inspect that output before reporting that execution started; a job ID alone is insufficient.

If output is missing or empty, inspect Slurm status, report pending or unconfirmed accurately, and check again after at least 20 seconds while awaiting confirmation. Do not resubmit merely because a job is queued. If Slurm reports a terminal failure before any output exists, report that failure rather than claiming a running job. For a short job already finished, report completed or failed based on logs and accounting instead of saying it is still running.

`STARTED` confirms the batch shell started; `ENV_READY` confirms virtual-environment activation. `RUNNING` marks application launch, and heartbeat messages every 10 seconds show that the batch shell remains active. A heartbeat does not prove training is making progress. Training code must also print and flush real progress regularly: setup stages, data loading, epoch/step, loss, evaluation, and checkpoint saves. Use `python -u` and, where appropriate, `print(..., flush=True)`.

Every new batch script must source `hpc_runtime.sh` before any workload command and retain the status/exit handling. Sync that helper along with the script. A missing environment must stop execution. Keep `.out` and `.err` in the same remote directory as the submitted script: the helper enforces this for scripts in `hpc/`; when adapting a script for another directory, update its absolute `#SBATCH --output` and `--error` paths and submit with that directory as `--chdir`. Do not use `$HOME` inside Slurm directives.

The heartbeat stops at exit. `COMPLETED` means the shell commands returned zero; inspect application metrics to confirm the intended result. Abrupt termination, such as a forced kill, can prevent final status printing, so also consult Slurm accounting. Each finite log check must close SSH afterward.

Use the `Invoke-Hpc` function above. Replace `212877` with the ID returned by your submission:

```powershell
Invoke-Hpc 'squeue -u alex116u1'
Invoke-Hpc 'scontrol show job 212877'
Invoke-Hpc 'sacct -j 212877 --format=JobID,JobName,State,ExitCode,Elapsed'
Invoke-Hpc 'cd ~/data/VirtualSink && tail -n 80 compare-212877.out'
Invoke-Hpc 'cd ~/data/VirtualSink && tail -n 80 compare-212877.err'
```

Current scripts use `<script-name>-<ID>.out` and `.err` beside the remote script. Job 212877 predates this convention and used `compare-212877.out` and `.err`. Log files may not exist until the job starts. Each check closes its SSH connection; avoid `tail -f` if you want the command to return immediately.

A pending job is waiting for resources or another scheduling condition. Inspect the reason before resubmitting. A job disappearing from `squeue` does not establish success: inspect accounting and logs. Accounting availability and update timing depend on the cluster configuration. See the [Slurm command overview](https://slurm.schedmd.com/quickstart.html).

Cancel a specific job only when intended:

```powershell
Invoke-Hpc 'scancel 212877'
```

If submission reports an invalid batch script, check the shebang and LF line endings. If the job fails after starting, read its error log for missing dependencies, incorrect paths, CUDA errors, or resource limits. Correct the cause before submitting again. Do not fall back to executing training on the local machine or login node.
