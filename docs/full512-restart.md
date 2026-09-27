# Full-attention 512-token restart (2026-09-13)

Supersedes full8192 training jobs 214466/214467 and their dependent jobs 214468/214469,
which were cancelled at the user's request. Existing SWA diagnostics 214465 still
use the immutable full8192 release and remain queued after the SWA pair.

New root: `~/data/smollm-full512-20260913`.

- Full causal attention, 512-token sequences, microbatch 1.
- 2048 sequences/update, 30 updates: unchanged 31,457,280 target tokens.
- Fresh matched seed 8 weights, data seed 42, sink std .02/bias +4.
- Validation: 64 sequences/update (32,768 tokens); final held-out split:
  2048 sequences (1,048,576 tokens). Data is regenerated with pinned sources.
- Window 512 streaming evaluation continues through 32k; positions beyond 512
  are explicitly marked as extrapolation. Eviction coincides with the trained
  length boundary, so this is not an isolated test of eviction within training length.

Jobs: benchmark 214471, data 214472, baseline 214473, query sink 214474,
comparison 214475, streaming diagnostics 214476.

K80 benchmark on comp001: baseline 1.9147 seconds/microbatch, query sink 2.0131.
Extrapolated training compute: 32.68 and 34.36 hours, respectively. Allow roughly
35-40 elapsed hours if both run concurrently, excluding queue delays and the
separate streaming diagnostic jobs. These are short synthetic-input benchmarks;
actual training metrics should replace the estimate after the jobs begin.
