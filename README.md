# VirtualSink

## Phase 3

The Qwen2.5 adaptation, token preparation and Slurm uptraining workflow are
documented in [Phase 3](docs/phase3.md). The implementation has CPU numerical
tests; pretrained uptraining awaits suitable compute and a prepared corpus.

## HPC execution

All training and GPU experiments run through Slurm on Bibalex HPC. See [the HPC guide](hpc/README.md) for setup, script authoring, submission, and execution confirmation.

From the repository root:

```powershell
.\hpc\submit_hpc.ps1 -Script test_gpu_access.sh
```

Commit and push changes before submission; the helper updates the remote checkout. The K80 environment is installed, but GPU access on `comp016` remains broken. See the guide before starting an experiment.
