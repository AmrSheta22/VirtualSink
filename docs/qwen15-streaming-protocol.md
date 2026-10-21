# Qwen2.5-1.5B streaming replication, 11 October 2026

## Fixed comparison

Use the completed seed-1729 attention-uptraining pair at the independently selected best checkpoint (step 975 for both arms), not last.pt at step 1000. Pin best.pt SHA256 values before validation and recheck them in every article task. Preserve the training model, tokenizer, dataset and source identities. Model Qwen/Qwen2.5-1.5B revision 8faed761d45a263340a0528343f099c05c9a4323. Training root `/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010`.

Evaluate the twelve existing CONFIRMATION_TITLES, without selecting by observed performance. These are the same WikiText articles used in the earlier Qwen0.5B study. This is a larger-model replication on that fixed set, not new-document confirmation. Each article supplies exactly 8,192 teacher-forced next-token predictions from its first 8,193 encoded tokens. Verify original source parquet SHA256 and record article text and token SHA256. No reset or recomputation of historical hidden states within an article.

Five policies, all with at most 512 physical K/V entries per layer including the current token:

1. Standard checkpoint with FIFO and absolute RoPE.
2. Virtual-sink checkpoint with FIFO and absolute RoPE.
3. Standard checkpoint retaining first four plus 508 recent tokens, absolute RoPE.
4. Standard checkpoint retaining first four plus 508 recent tokens, cache-relative rebased RoPE.
5. Virtual-sink checkpoint with sinks disabled, FIFO and absolute RoPE.

The virtual sink occupies no physical cache slot. Rebasing caches raw keys and reapplies RoPE at retained cache indices 0 through 511; the current query uses the final cache index. This is an explicitly defined rebased prefix baseline, not a claim of reproducing every official StreamingLLM implementation detail.

## Primary endpoint and report

For every article, average CE at zero-based input positions 512 through 8191 (7,680 targets), then calculate standard-FIFO minus virtual-FIFO CE. Positive favors virtual sinks. Resample twelve paired articles 10,000 times, seed 1729, for a percentile 95% interval; transform mean CE differences into PPL ratios. Report article wins and worst article. Do not resample individual correlated tokens as independent observations.

Secondary descriptive comparisons: absolute/rebased prefix controls, disabling sinks, pre-eviction positions 0 through 511, and beyond-ten-window positions 5120 through 8191. No multiple-comparison-corrected significance claim for secondary endpoints.

The aggregator independently recomputes region means from raw per-token CE, cross-checks CE and input token IDs against diagnostic NPZs, and verifies policy metadata and cohort source/checkpoint consistency. Require every article and all five policies; otherwise write INCOMPLETE.json and withhold a final conclusion. Total 491,520 scored predictions, including 460,800 post-eviction predictions.

Only one seed pair was completed. Article intervals do not represent training-seed uncertainty or cross-corpus generality.

## Validation and execution

Local suite: 19 tests passed, including a Qwen GQA/bias/RoPE rebasing oracle after repeated eviction. The oracle independently stores absolute-rotated keys, then applies delta rotations to current cache indices; production stores raw keys.

Compute validation repeats CPU tests in the legacy runtime, compares real-checkpoint cached versus parallel logits across eviction for both enabled/disabled model variants, and verifies rebased-prefix logits against the independent oracle over 1,153 tokens, crossing two eviction windows. Every article task repeats real-checkpoint cached/parallel checks and rebased-prefix oracle checks over 641 tokens.

Slurm jobs: pin 216812; validation 216813; article array 216814 indices 0–11, concurrency 4; aggregate report 216815. Validation depends on successful pinning, article tasks on successful validation; report follows termination of the array and rejects incomplete results. GPU tasks use one K80, exclude comp013/comp016, request 32 GB host RAM and six hours per article. No automatic retries or modifications to training checkpoints.

Remote evaluation root: `/cluster/users/alex116u1/data/qwen15-streaming-20261011`.

Local deployment provenance: `artifacts/qwen15-streaming-20261011/`.
