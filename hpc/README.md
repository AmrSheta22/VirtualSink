# HPC

## Phase 3 K80 update (2026-09-09)

GPU probe 214314 completed successfully on comp008 with PyTorch 1.12.1+cu113
and a K80 (11,441 MiB). The older failures below remain relevant to comp013/
comp016, which the Phase 3 jobs exclude. See [the K80 pilot protocol](../docs/phase3-k80-runs.md).
The pilots use isolated committed release checkouts and a separate environment.
GitHub publication was not approved; the release was transferred directly as a
Git bundle to the authorized HPC account, preserving both dirty checkouts.

All training and GPU tests run through Slurm, never locally or on the login node.

## Run

From local PowerShell, after committing and pushing your changes:

```powershell
.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh
# Recheck without submitting again:
.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh -JobId 123456
```

For manual submission in an HPC SSH session:

```bash
cd ~/data/VirtualSink
git pull --ff-only
cd hpc
sbatch test_gpu_access.sh
exit
```

If pulling fails, resolve the remote changes first. Do not submit an outdated script or blindly reset local edits. The helper stops on update failure. Local uncommitted files are not transferred by Git.

## Check

Wait **at least 20 seconds** after submission, then read `hpc/test_gpu_access-<job-ID>.out` and `.err`. The PowerShell helper does this and closes SSH after each command. For manual submission, reconnect to read the files and close SSH afterward.

Users and LLMs must inspect actual output before confirming execution. Missing output means unconfirmed or queued; check `squeue -u alex116u1` and retry later without resubmitting. Use `sacct -j <job-ID> --format=JobID,State,ExitCode` for completion. A job ID alone does not mean the GPU works.

## Environment and current issue

- Login: `alex116u1@login02.c2.hpc.bibalex.org`
- Checkout: `~/data/VirtualSink`
- Environment: `venv-k80` — Python 3.9, PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113, NumPy <2.
- Credentials: local `server_access/`, excluded from Git.
- Logs stay beside their batch scripts in `hpc/`.

The compatible PyTorch environment is installed. Jobs 212885 and 212886 still report no CUDA GPUs; the downgraded runtime alone did not fix device access. Earlier diagnostics on `comp016` also failed in `nvidia-smi`. Scripts exclude that node and retain the existing `comp013` exclusion; other nodes may have long queues. A successful GPU test is still required before training. Ask the HPC administrator about the GPU handle error on `comp016`.

## Write a job

Module check (job 212889, comp016): no module, `CUDA/11.1.1-GCC-10.2.0`, and `CUDA/11.7.0` all failed identically. Both allocated GPUs failed individually; `cuInit` returned 100 and `nvidia-smi` could not obtain either device handle. Loaded kernel driver: 470.256.02. The probe completed, but none of its CUDA tests passed. Logs remain in remote `hpc/module_probe-212889.out` and `.err`. Loading a toolkit module did not resolve device access; give these details to the HPC administrator. The [live site guide](https://hpc.bibalex.org/quick-start) was used because the supplied archive snapshot could not be fetched.

Copy `test_shell.sh`, change the job name, time limit, and log filenames, then replace its final echo with your real `python -u ...` command. Keep the Slurm guard, environment activation, heartbeat, and exit reporting. Print real training progress (steps, loss, checkpoints) with flushing; a heartbeat only means the shell is active. Save shell files with LF endings.

Submit the copy with `-Script your_script.sh`. Keep all `#SBATCH` directives before executable commands and use absolute log paths. Do not change shared code while jobs are queued or running.
