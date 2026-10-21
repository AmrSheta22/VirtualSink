# Qwen0.5B checkpoint speed benchmark

Experiment `qwen05-checkpoint-speed-20261016-r1`, Slurm job 217214, compares
the existing recurrent implementation with `OptimizedRecurrentSinkLM` on the
actual ordinary-memory Qwen2.5-0.5B checkpoint. Both arms execute sequentially
in one Slurm allocation on comp002, one K80, four CPUs, 64 GB host memory.
Runtime is PyTorch 1.12.1+cu113, FP32. The running production job is untouched.

At submission, the latest checkpoint was step 1088, saved on 16 October 2026
at 09:45:06 Cairo time, after 542,998 input tokens. Its cursor is article 121,
offset 6400. A hard link pins that immutable checkpoint inode on the cluster;
the production trainer saves to a different temporary inode and atomically
replaces its own `last.pt`, so later saves cannot change the benchmark copy.
No checkpoint is downloaded to the local machine.

Each arm independently restores exactly those model weights, AdamW moments,
recurrent/cache states for both streams, LR phase, data cursor and RNG states.
Both call the original `train_pair` routine with its unchanged production
configuration: two causal passes per update, two connected 256-token segments,
cache capacity 128, layer and vocabulary-loss checkpointing, vocabulary-loss
chunk 8, gradient clipping at 1 and full-parameter AdamW with foreach disabled.
Only training data are accessed. No benchmark model updates are saved back.
The recurrent reference, native backbone, data and training helpers have
identical hashes to the frozen production release.

Each arm executes six updates: one warmup at 1089, followed by five measured
updates 1090–1094. The first four measured updates each have 512 input tokens;
the final update ends the current article and has 429 tokens. Comparisons
use matched cursors and token counts, and separately report the four full
updates. CUDA synchronization surrounds forward/backward, clipping and the
AdamW update; loading, diagnostic copies, printing and parity checks are
outside the timer. Tokens/second counts input tokens once despite both passes.

The final audited timing excludes matched updates 1090 and 1091 in both arms:
an overlapping read-only GPU/process diagnostic occurred during reference
1090, and a debugger briefly paused reference 1091. This exclusion was recorded
before any optimized-arm measurements. Primary timing therefore uses clean
matched steps 1092–1094 (three updates), and separately the two full 512-token
updates 1092–1093. Raw five-sample summaries remain available for transparency.
The debugger was detached; the production job was never inspected this way.

After the first update, all clipped parameter gradients are compared by
absolute error and global relative L2 error. For every update, both stream
states (cache K/V, LSTM cell and exposed state), training losses and gradient
norm are checked. A failed check stops the job before a successful comparison
is emitted. The earlier CPU tests also check unscaled gradient parity.

The optimized subclass batches gate-input projections, physical attention,
memory reads and value aggregation; the LSTM scan remains sequential. Its
masked attention score matrix has O(T*(T+C)) temporary storage. A measured
improvement at T=256 does not establish scalability to unbounded segments.

Small collected artifacts, source manifests and Slurm allocation are under
`artifacts/qwen05-checkpoint-speed-20261016-r1`. The checkpoint and gradient
tensors stay on the cluster. This is a short implementation benchmark from
one checkpoint, not a test of additional learning or generalization.

## Completed result

Job 217214 completed with exit code zero, no restarts and an empty error log
at 12:17:04 Cairo time on 16 October 2026. Both arms used the same K80 in one
allocation. The reference reproduced the original production log's CE and
gradient norms exactly for all six replayed updates 1089–1094.

Audited comparison uses clean matched updates 1092–1094. The first two have
512 input tokens, the last 429. Their speedups are respectively 5.254x,
5.226x and 5.171x. Across all three, time falls from 155.214 to 29.737 seconds
per update: **5.220x throughput and 80.841% less update time**. Input throughput
is 3.1204 versus 16.2871 tokens/second.

| Full 512-token updates only (1092–1093) | Current | Optimized |
|---|---:|---:|
| Mean seconds/update | 164.0697 | 31.3104 |
| Speedup | — | 5.240x |
| Peak allocated CUDA memory across all six updates (GB, decimal) | 9.5953 | 9.5985 |

The full model therefore retains a roughly **5.2x** implementation speedup,
compared with the earlier dummy model's 5.82x. GPU memory use is essentially
unchanged at this shape; the dummy-model memory reduction did not transfer.

Numerical checks passed: maximum CE difference across all six updates was
4.7684e-7. First-update clipped gradient relative L2 error was 6.8336e-6,
with maximum absolute error 2.6822e-7. Both stream states were checked after
each update with atol 3e-4, rtol 1e-3; the maximum absolute difference among
all state tensors was 0.0010986. This is FP32 numerical agreement, not bitwise
identity. No validation/test data or production writes were involved.

Primary result: `artifacts/qwen05-checkpoint-speed-20261016-r1/audited-comparison.json`.
Raw results including the diagnostic-contaminated timings remain in
`comparison.json`; exclusions are recorded separately in
`timing-exclusions.json` and must be applied before quoting the speedup.
The primary sample is only three paired updates from one checkpoint; it
does not establish full-run convergence equivalence or change the ongoing
training job's expected finish time.
