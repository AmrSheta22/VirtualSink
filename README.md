testing rse 

## HPC execution

All training and GPU experiments run through Slurm on Bibalex HPC. See [the HPC guide](hpc/README.md) for setup, script authoring, submission, and execution confirmation.

From the repository root:

```powershell
.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh
```
