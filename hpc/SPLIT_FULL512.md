# Full-512 replication split across Bibalex and MOHESR

**Current assignment: all six arms now run concurrently on Bibalex.**
Seed 8 uses jobs **216537_0** (standard) and **216537_1** (query_sink), with
comparison job **216538**. Seeds 9/10 keep their original four jobs unchanged.
MOHESR job 11593 was stopped after saving a checkpoint at update 1, microbatch
79/128. A preserved hard link named `migration-checkpoint.pt` remains in its
standard-arm directory. Both final seed-8 arms restart fresh under Bibalex's
runtime, giving all six final arms the same K80/PyTorch 1.12.1 environment.
The earlier mixed-cluster assignment below is retained as execution history.

The collector reads `artifacts/split-full512-20261009/assignment.json` for the
current assignment and no longer contacts MOHESR for active progress. The
three-hour report automation now covers all six Bibalex arms. The new pair
passed 27 tests (2 optional cases skipped), verified the dataset and passed
full-model GPU checks; both arms' shared initialization hashes matched.

The original six-arm replication is now assigned as follows:

| Cluster | Job | Seed | Arm |
| --- | --- | --- | --- |
| MOHESR | 11593, sequential first | 8 | standard |
| MOHESR | 11593, sequential second | 8 | query_sink |
| Bibalex | 216528_0 | 9 | standard |
| Bibalex | 216528_1 | 9 | query_sink |
| Bibalex | 216528_2 | 10 | standard |
| Bibalex | 216528_3 | 10 | query_sink |

The four Bibalex tasks run concurrently on four K80 devices across comp001 and
comp002. Comparison jobs 216529 (seed 9) and 216530 (seed 10) wait for their
respective pairs. Use array-task IDs to identify jobs: Slurm also assigns
different internal numeric job IDs to array tasks.

MOHESR saved its in-progress gradients at update zero, microbatch 39/128, before
requeueing the same job. Its launcher now reads `arm-indices.txt` containing
`0 1`, so it resumes the standard arm and then runs the query-sink arm; seeds
9 and 10 are excluded. No optimizer progress was discarded.

## Matched recipe and limitations

All six arms use the unchanged numerical source from the original MOHESR
release, original SHA256-verified data and ordering, full causal context 512,
FP32, microbatch 16, global batch 2048 sequences, chunk 32, and 30 optimizer
updates. Sink initialization and evaluation budgets remain as documented in
[MOHESR_FULL512.md](MOHESR_FULL512.md). No auxiliary loss is used.

Bibalex K80s require its verified PyTorch 1.12.1+cu113 environment; MOHESR uses
PyTorch 2.6.0+cu124 on an A100 MIG slice. Each seed pair uses one cluster/runtime
and strict configuration/initial-weight hash checks. Across seeds, initialization
seed is confounded with cluster/runtime. Report pairwise effects and disclose
this limitation when aggregating; this is not a same-hardware three-seed study.

Before deployment, source hashes were checked against the frozen MOHESR release.
All four Bibalex tasks passed **27 tests** (2 Transformers-dependent cases
skipped), verified all original data hashes, and passed full-model batch-16
forward/backward/Adam checks. Within-seed shared initialization hashes matched
for both arms of seeds 9 and 10. Current error logs were empty at launch.

## Locations and reports

- MOHESR root:
  `/nfs/slurm/alex005/VirtualSink/experiments/smollm-full512-20261009`.
- Bibalex root:
  `/cluster/users/alex116u1/data/smollm-full512-replication-20261009`.
- Each root records authoritative IDs in `jobs.tsv` and the assignment in
  `split-assignment.json`.
- Local assignment and snapshots:
  `artifacts/split-full512-20261009/` (Git-ignored).

Run the read-only collector:

```powershell
.\venv\Scripts\python.exe hpc/report_split_full512.py
```

It reads both queues/accounting, each arm's metrics and summaries, and completed
pair comparisons. It writes `latest-status.json` locally and, once all three
pairs finish, `aggregate.json`. That aggregate includes the mixed-runtime
limitation and does not pool test blocks across seeds.

The existing three-hour report automation now covers **both clusters** and
pauses after the combined final report. Local reports require the computer and
desktop app to remain running; Slurm training proceeds independently.

Do not rerun deployment against these active experiment roots. Both launchers
checkpoint partial accumulation and requeue on their allocation limits.
