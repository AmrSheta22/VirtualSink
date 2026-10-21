# Recurrent-memory implementation benchmark

Experiment: `recurrent-speed-20261016-r1`. Current implementation job 217212;
optimized implementation job 217213, dependent on successful completion of
217212. Both request one K80 on comp002, four CPUs and 16 GB host memory.
The ongoing Qwen long uptraining release and jobs are untouched.
The reference recurrent implementation, native backbone and training-loss
helper have identical SHA256 hashes to the frozen Qwen long uptraining release.

Both arms use the same random Qwen-style dummy model (four layers, hidden
128, intermediate 384, four query/two KV heads, vocabulary 2048), seed,
synthetic tokens, AdamW settings and FP32 computation. Each optimizer update
has two equally weighted causal passes, each with two connected 256-token
segments, physical cache 128, layer/vocabulary-loss activation checkpointing,
and recurrent states detached after each update. States reset at 8192 tokens.
The main statistic excludes five warmup updates and includes 30 timed updates;
CUDA synchronization surrounds forward, backward, clipping and optimizer work.
The throughput numerator is 512 input tokens/update, although both passes
execute, matching the accounting of the long uptraining experiment.

The optimized subclass batches input gate projections, null and memory reads,
physical attention, and value aggregation. The LSTM state scan remains causal
and sequential. Model parameters, recurrent equations, RoPE positions, cache
eviction order, residual connections and MLPs retain the reference semantics.
The benchmark sets appreciable initial memory mass in both implementations.

The attention implementation uses an explicitly masked segment score matrix:
temporary attention storage is O(T*(T+C)), rather than the reference loop's
per-token bounded attention workspace. This tradeoff is appropriate to test at
256-token segments, but does not establish suitability for arbitrarily large
segments or predict Qwen0.5B throughput.

Local parity: 24 tests cover two-segment forward states, all parameter
gradients, connected gate diagnostics and checkpoint recomputation, across
cache capacities 1/3/20 and read/write/teacher/value/forget/bypass switches.
Each Slurm job repeats those tests and checks CUDA training loss, gradient
norm, parameter gradients and returned states before timing. Any failed check
prevents a successful result. Source SHA256 manifests, logs and timing samples
are retained under the experiment's artifacts directory.
The existing recurrent/teacher/training/long-run CPU tests also pass (23 tests).
These synthetic repeated tokens measure implementation cost; their training
losses do not establish language-model quality or long-memory generalization.

## Measured result

Both jobs completed with exit code zero on 16 October 2026: current at
11:43:37 Cairo time; optimized at 11:46:51. Slurm assigned both comp002 GPU
index 0 and CPU IDs 0–3, with no restarts. Runtime: PyTorch 1.12.1+cu113.

| Metric | Current (217212) | Optimized (217213) |
|---|---:|---:|
| Mean seconds/update | 25.4288 | 4.3701 |
| Median seconds/update | 25.4078 | 4.4375 |
| Input tokens/second (512/update) | 20.1346 | 117.1601 |
| Peak allocated CUDA memory (MiB) | 62.0493 | 32.5850 |

Mean throughput improved **5.82x**, reducing update time **82.81%**.
Median-based speedup is 5.73x. Three successive matched ten-update windows
give speedups 5.84x, 5.85x and 5.76x. Timing includes forward, checkpoint
recomputation/backward, gradient clipping and AdamW, excluding startup,
parity checks, five warmups and printing. It includes two document resets
in the measured schedule, identically in both arms.

GPU parity passed in both jobs: largest returned-state absolute error
3.8147e-6 and largest clipped-parameter-gradient absolute error 2.2352e-8.
The maximum difference in summed two-pass training losses across 30 matched
updates was 1.5259e-5. Local 24-case tests separately verify unscaled gradients
before any clipping. Error logs are empty. No checkpoints or model downloads
were needed.

This is one sequential run per implementation on a 1,115,344-parameter dummy
model, with 30 timing samples per run; it does not establish a 5.82x speedup
for Qwen0.5B. Larger dense projections, vocabulary heads and memory matrices
can change the bottleneck. The measured memory reduction also applies only
to this shape; the batched attention workspace still grows quadratically in
segment length. The optimization has not been applied to ongoing Qwen jobs.

Raw samples, exact configuration, Slurm records and comparison arithmetic:
`artifacts/recurrent-speed-20261016-r1/{current,optimized,comparison}.json`
and `slurm-status.txt`; source hashes: `source-sha256.json`.
