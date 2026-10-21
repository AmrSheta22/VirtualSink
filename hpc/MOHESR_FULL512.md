# SmolLM full-512 replication on MOHESR

**Current assignment:** four arms now run in parallel on Bibalex; only the
seed-8 standard/query-sink pair runs consecutively on MOHESR. See
[SPLIT_FULL512.md](SPLIT_FULL512.md) for current job mapping and reporting.
The six-arm sequential assignment below describes the initial deployment.

The native decoder previously detached training gates, preventing auxiliary-loss
gradients from reaching sink and query parameters. `Layer.forward` now retains
the gradient path. Four regression cases exercise the auxiliary objective alone:
full and sliding-window attention, each with and without checkpointing.

This replication uses **no auxiliary loss**, matching the earlier scratch
experiment. It tests reproducibility across initialization seeds rather than
the effect of regularization.

## Fixed protocol

- Standard and query-sink arms for model seeds 8, 9, and 10.
- SmolLM 135M from random recipe initialization; no pretrained weights.
- Full causal attention, context 512, 30 optimizer updates, global batch 2048
  sequences (1,048,576 input tokens per update).
- FP32, AdamW and the existing shortened learning-rate schedule. The first
  update has zero learning rate, as in the original experiment.
- Sink initialization: weight standard deviation 0.02 and bias 4; chunk 32.
- Identical original token data and ordering from Bibalex, data seed 42.
  Manifest SHA256:
  `0d8cdb9cb8a074864e481bdec43c5e33f04f69bb2ca01fc0f0224f4baa121cdb`.
- Microbatch **16**, selected by a Slurm-only benchmark and used for all six
  arms. The global batch and token order are unchanged; floating-point
  accumulation grouping differs from the original microbatch-1 experiment.
- Validate on the same 64 blocks every update. Compare fixed final checkpoints
  on all 2048 test blocks; reevaluate trained query-sink models with sinks
  disabled. Preserve within-seed paired bootstrap intervals and aggregate
  initialization-seed gains separately.

Existing tests verify equivalent accumulated gradients across microbatch
groupings and exact recovery of partial accumulation through checkpoints.
Pair comparisons require matching initialization hashes, data/source hashes,
and training settings.

## Execution

Remote root:
`/nfs/slurm/alex005/VirtualSink/experiments/smollm-full512-20261009`.
Use its frozen `release/src` through `PYTHONPATH` and the verified
`/nfs/slurm/alex005/VirtualSink/venv-modern` environment.

Verification job **11591** completed successfully: **123 tests passed**, all
original data hashes verified, and both full-model GPU optimizer checks passed.
Sequential pipeline job **11593** benchmarks batches, trains the six arms,
compares each completed pair, and writes `aggregate.json`.

The account's `hpc_qos` permits only one submitted and one running job.
Consequently, all arms and comparisons run within one allocation. The Slurm
launcher forwards preemption signals to the trainer, saves partial gradients,
and requeues the same job across 24-hour allocations. Completed arms are
skipped when the job resumes. `jobs.tsv` is authoritative for job IDs.

The first pipeline, job 11592, stopped during its microbatch-32 benchmark with
a PyTorch MIG allocator NVML assertion, before creating any training outputs.
The replacement benchmarks sizes 8 and 16 and treats that specific allocator
error as a failed candidate. Any other runtime error still stops the pipeline.

The benchmark initially measured about 50 training hours per arm at microbatch
1. The selected microbatch 16 measured 25.63 seconds for the standard arm and
25.94 seconds for the query-sink arm, estimating 27.34 and 27.67 training hours
per arm. The six sequential arms therefore require roughly 165 training hours
plus evaluation and checkpoint I/O. Read `microbatch-benchmark.json` and
`selected-microbatch.txt` for evidence and actual training metrics for updated
estimates.

Training execution was confirmed on job 11593: seed-8 standard completed its
initial validation (CE 11.31127), saved `best.pt` and `last.pt`, and performed
the first of 128 accumulated microbatches for update zero. Its first backward
pass took 25.90 seconds with finite CE 11.26383. The current error log was empty.

## Reports

The active chat automation `mohesr-training-report-every-3-hours` reports every
three hours and pauses after the final aggregate report. Local scheduled reports
require the computer to remain on and the desktop app running. Training itself
runs independently on Slurm.

Run a compact read-only status report from the local workspace:

```powershell
.\venv\Scripts\python.exe hpc/mohesr_remote.py '/nfs/slurm/alex005/VirtualSink/venv-modern/bin/python /nfs/slurm/alex005/VirtualSink/experiments/smollm-full512-20261009/release/hpc/report_mohesr_full512.py /nfs/slurm/alex005/VirtualSink/experiments/smollm-full512-20261009'
```

Inspect `logs/pipeline-11593.out` and `.err` for current errors. Each arm stores
`config.json`, `metrics.jsonl`, resumable `last.pt`, final checkpoint and held-out
summary in `seeds/seedN/runs/VARIANT`. Comparisons are under
`seeds/seedN/comparison/`.

`deploy_mohesr_full512.py` documents deployment to a fresh experiment directory:
it refuses to overwrite an existing root, transfers no credentials, validates
local data hashes, and submits a single pipeline job. Do not rerun it against
this already-running experiment.
