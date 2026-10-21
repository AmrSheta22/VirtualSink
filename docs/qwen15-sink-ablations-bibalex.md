# Qwen1.5B sink experiments on Bibalex â€” 11 October 2026

User-authorized follow-up to the completed seed-1729 pair. Two fresh runs start from the pinned Qwen2.5-1.5B pretrained weights at revision `8faed761d45a263340a0528343f099c05c9a4323`, not the earlier trained checkpoints. Training uses the same verified WikiText token streams, seed 1729, attention-projection updates only, FP32/K80, 1,024-token blocks, window 512, accumulation 8, microbatch 1, and 1,000 optimizer steps (8,192,000 input tokens). The remaining pretrained parameters are frozen.

## Exact arms

| Run / job | Architecture | Auxiliary loss | Peak learning rates | Schedule |
|---|---|---|---|---|
| `highlr-single` / 216867 | One query-conditioned zero-value sink per query head | Anti-collapse, gamma 0.01 | Pretrained Q/K/V/O: 2e-5; sink score weights and biases: 5e-4 | Warmup 100; smoothly reduce sink/base ratio from 25 to 1 over steps 100â€“300; same cosine decay thereafter |
| `highlr-four` / 216868 | Four query-conditioned sinks per query head | None, gamma 0 | All trainable parameters: 1e-4 | Warmup 100, shared cosine decay to step 1,000 |

The distinct schedule is applied only to the auxiliary-loss run, as the user's final correction requested. Both arms use AdamW, global gradient clipping at 1, pretrained projection weight decay 0.01, and zero sink weight decay. Equal learning rates do not imply equal effective updates under AdamW. The first optimizer update uses the existing trainer's zero-rate warmup start; step-indexed logged rates are those prepared for the next update. At scheduler step 300, the common rate is approximately 1.7660e-5, and both reach zero at step 1,000.

For four sinks, score parameters are independent per head and slot. The first value is constructed as an exact zero tensor and cannot be trained. The other three vectors are trainable, initialized with RMS scales `[1/3, 2/3, 1] * 1e-9` relative to the pretrained V-projection weight RMS of their layer. Random vector directions are normalized to unit RMS. These are initial magnitudes, not constraints on subsequent trained values. Sink score weights have standard deviation 1e-4. Each bias is `-3.5-log(4)` to preserve aggregate initial virtual logit rather than quadruple virtual mass. This is a value-vector architecture, not an extra query-conditioned value projection. Query conditioning enters each sink's score.

The earlier one-sink campaign already used anti-collapse gamma 0.01; this first run tests the higher sink rate and converging schedule. Differences between the two new arms cannot be attributed solely to sink count because objective and schedule also differ.

## Frozen deployment and execution gates

Remote root: `/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2`.

Local provenance: `artifacts/qwen15-sink-ablations-20261011-r2/`, including plan, source hashes, frozen release, batch scripts, jobs.tsv and observations. Remote jobs.tsv is authoritative. Earlier training and streaming releases remain isolated.

- K80 validation gate 216866 runs 41 tests, full-model auxiliary-only gradient checks, two accumulated peak-rate optimizer steps per arm, finite connected language gradients, controlled nonzero value gradients for every vector, actual AdamW value updates and exact-zero checks, then 641-token streaming/SWA parity including eviction.
- Separate training jobs 216867 and 216868 depend on successful gate completion and can run concurrently.
- CPU checkpoint pinning 216869 depends on **both** training jobs succeeding, checks trainer completion summaries and hashes their validation-selected best checkpoints plus the existing standard checkpoint.
- GPU streaming parity 216870 depends on pinning. It verifies full/window parity for all three models and independent rebased-prefix parity for the standard control.
- Streaming array 216871 has twelve articles, concurrency four, and depends on parity success. Each scores 8,192 predictions per article with 512 physical cache entries. Policies: standard FIFO, standard prefix-four, standard rebased prefix-four; each new model with sinks enabled and disabled under FIFO. Aggregate sink-score diagnostics are logsumexp over all sink logits; virtual mass is the sum of their probabilities.
- CPU report 216872 runs after any termination of the article array, requires all twelve units and seven policies, verifies token diagnostics, reaggregates losses and computes 10,000 paired article-bootstrap resamples. Missing units produce INCOMPLETE.json and no final result.

Training allocations are 48 hours, with optimizer-boundary checkpointing and requeue after a successful pause. Resume requires exact source/config/data equality. Validation every 25 steps uses 32 packed blocks, selects lowest CE, and performs the separate 32-block trainer test. Packed-block tests do not establish persistent streaming behavior.

Streaming repeats the fixed twelve WikiText articles from the earlier study. One training seed and article-bootstrap uncertainty do not establish cross-seed or cross-corpus significance. The original standard control is reused, with its selected step-975 checkpoint; each new arm's best step is determined by validation rather than assumed in advance. No intentionally cancelled seed is restarted.

The three-hour monitor follows this campaign and the already-submitted original streaming campaign at `/cluster/users/alex116u1/data/qwen15-streaming-20261011` (216812â€“216815), reporting elapsed training separately from queue time, observed throughput, approximate remaining time and failures. It stays active until both campaigns finish or fail terminally.


## Repair and retry provenance — 11 October 2026

The first release at `/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011` failed its smoke gate 216820 after 6m19s. A check incorrectly required every head/slot to receive a nonzero language-loss gradient on each batch. Training jobs 216821/216822 and dependent pin/parity/streaming jobs were automatically cancelled before execution. Their logs and source are preserved; neither run produced a training checkpoint.

Diagnostic allocation 216864 completed in 2m02s. On the first eight microbatches, all three trainable values in layer0/head0 had zero language gradients; the remaining head/slots had nonzero gradients. Every one of the 28 layers, 12 heads and three trainable slots (1008 vectors) received a strictly positive gradient in a controlled attention probe. The test failure did not establish a disconnected value parameter. A saturated-softmax regression also demonstrates legitimate zero language gradients despite a working value gradient path.

The repaired gate requires every value parameter to belong to the optimizer, receive a finite connected gradient, pass a controlled nonzero-gradient probe for each head/slot, and produce actual aggregate AdamW value updates. It records zero language-gradient slots rather than rejecting them. Missing checkpoint identity now produces INCOMPLETE.json with an explicit reason. Local suite: 41 passed. Experimental architecture, initial values, learning rates, objectives and token budget are unchanged. Replacement jobs and frozen source use the `-r2` root listed above.

Replacement gate 216866 completed successfully in 7m55s. Cluster suite: 39 passed, 2 optional-backend tests skipped. Both arms passed two accumulated optimizer steps. Streaming parity max absolute logit error: single 0.001313, four 0.000718 (tolerance atol0.003/rtol0.0001). Four-sink values changed under AdamW, and the fixed zero slot remained exactly zero. Peak reserved memory: 8,659,140,608 bytes single; 8,661,237,760 bytes four. Detailed smoke results are saved locally in the r2 artifacts folder.

Both replacement trainers started at 21:13:18 Cairo on 11 October 2026. Startup logs show successful native model loading, 8,192,000-token budget, and empty stderr. Runtime config files match the requested distinct auxiliary-arm schedule and four-sink shared learning rates. Initial validation is underway; optimizer throughput is not yet measured.
