# MOHESR VirtualSink setup

SSH connects as `alex005` to `hpclogin.mohesr.gov.eg` with the local,
Git-ignored files `server_access/private_mohesr` and
`server_access/password_mohesr.txt`. Commands must continue through `manage2`:
the login2 gateway's Slurm commands return a protocol authentication error.
The interactive shell normally performs this redirect automatically.

The modern environment is `/nfs/slurm/alex005/VirtualSink/venv-modern`.
Its Python comes from the isolated Miniconda installation at
`/nfs/slurm/alex005/envs/mohesr-python312` (Python 3.12.11).
Bibalex's modern environment uses Python 3.12.3; both use PyTorch
2.6.0+cu124, torchvision 0.21.0+cu124, and NumPy 2.5.2.
MOHESR has no matching environment modules; the PyTorch wheels provide
the CUDA 12.4 runtime libraries.

Current source is copied to `VirtualSink/releases/20261009` and installed
editable into `venv-modern`. The older checkout is preserved. Project extras
include experiments, dev, and uptraining (transformers 4.51.3).
The reference and installed package lists and installation log are under
the remote `VirtualSink/hpc` directory.

Install Paramiko locally if needed (`python -m pip install paramiko`), then:

```powershell
python hpc/mohesr_remote.py 'sbatch /nfs/slurm/alex005/VirtualSink/hpc/gpu_specs_mohesr.sh'
python hpc/mohesr_remote.py 'squeue -u alex005'
python hpc/mohesr_remote.py 'sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed'
```

Read both job logs at least 20 seconds after submission. The diagnostic runs
a CUDA forward/backward operation and the project tests through Slurm.
Requesting `--gres=gpu:1` in `gpu5` currently allocates an A100 MIG slice,
not a whole physical GPU. Keep computational workloads inside Slurm.

## Verified 2026-10-09

Job **11590** completed with exit code **0:0** on `gpu5srv` in 31 seconds.
Python reported NVIDIA A100 80GB PCIe MIG 1g.20gb, 19.5 GiB memory,
compute capability 8.0, 14 multiprocessors, CUDA runtime 12.4, and
CUDA driver API version 12.8. The GPU forward/backward check passed,
and all **119 project tests passed** in 25.26 seconds.
Logs: `VirtualSink/hpc/virtualsink-gpu-specs-11590.out` and `.err`.
