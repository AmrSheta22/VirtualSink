# Evicted-block memory pilot — 18 October 2026

User approved replacing literal LSTM memory with explicit compression roles.
Implementation: `src/virtualsink/uptraining/block_memory.py`. Existing recurrent
and static-sink backends are unchanged.

## Architecture and semantics

Each layer/head has one shared learned affine null score and an identically
zero null value. Every completed block of evictions publishes three K/V slots:

1. Long-term: arithmetic mean of unrotated keys and values.
2. Short-term: mean weighted by attention received during each evicted token's
   physical-cache lifetime. Include its own query; add epsilon 1e-8 to each
   accumulated weight, giving a uniform fallback if all attention is zero.
3. Free memory: a learned per-head query attends to evicted unrotated keys;
   use its softmax weights to pool both keys and values. Streaming log-sum-exp
   accumulators reproduce block softmax without storing the evicted block.

Physical tokens use absolute RoPE. Memory uses unrotated queries/keys without
RoPE. Physical, null and published memory logits share one softmax. Per-role
read biases start at -8; values retain ordinary magnitude. No initial read-mass
calibration or guaranteed near-identity initialization is claimed.

At token position t (zero based), eject t-capacity before attention, accumulating
that token's received attention from earlier queries. Publish after the last
token of each eviction block, and allow the current query to read that block.
The current query cannot contribute importance to an already evicted token.
No partial block can be read. For capacity=block_size=128 the first block is
published at position255, and the number of content slots after N input tokens
is `3*floor(max(0,N-128)/128)`. Importance receives no future-query information.

The bank grows with document length; there is no silent replacement or merging
of older blocks. Mean pooling is lossy and does not preserve arbitrary facts or
ordering. Physical importance is per query head even with GQA; summary K/V is
also per query head. This memory is larger than the GQA physical cache per slot.
Accumulator state stores six D-vectors and three scalars per layer/head, plus
one importance scalar per cached token. Published summaries store two D-vectors
per slot. Total-state-budget comparisons remain future work.

State is returned functionally, resets with state=None, detaches at TBPTT
boundaries, and preserves complete forward article history. Layer checkpointing
passes every state tensor explicitly. This FP32 reference supports aligned
unpadded Qwen/Llama streams, not independent per-row resets or rebased positions.

## Correctness gate

26 local tests passed (13 new block tests and 13 existing recurrent tests).
Coverage includes independent closed-form pooling and lifetime-importance
calculations, publication timing, capacity1, different block/window sizes,
disabled parity with native sliding attention, tokenwise/segmented parity,
causality, batch isolation, reset, no mutation, detached physical-cache storage,
delayed gradients, checkpoint output/state/gradient parity, role/value
interventions, checkpoint architecture rejection, extreme pooling scores, and
full-article evaluation token accounting.

The same26 tests passed under the cluster runtime in5.04 seconds. An additional
local tiny-model end-to-end runner check executed both arms for two updates,
crossed document boundaries, saved checkpoints, and ran every evaluation
condition. Both arms processed538 predictions with cursor[2,0]; their pure-FIFO
reports agreed exactly. This tests runner plumbing, not model quality.

The Slurm gate reruns these tests using the cluster PyTorch runtime, loads pinned
pretrained Qwen2.5-0.5B, checks 65-token/tokenwise streaming parity at window16
with appreciable memory reads, then performs two memory-only AdamW updates.
It requires finite nonzero gradients for every new parameter and an actual
learned-pooling-query update in every layer. No quality claim follows from this.

GPU gate217401 completed successfully on Tesla K80 / PyTorch1.12.1+cu113:
maximum streaming logit error0.00034356,44,352 new parameters, and finite nonzero
gradients for every new parameter. Both AdamW steps updated the learned pooling
query in all24 layers, taking10.28 and9.93 seconds. Peak reserved GPU memory was
2,703,228,928 bytes (about2.52 GiB). Average attention mass at the gate's deliberately
stronger read biases was null0.08123, mean0.00936, importance0.01696,
free0.00937. This short window16 arithmetic probe is not the pilot's window128
initialization and is not a model-quality result. Canonical report:
`artifacts/block-memory-qwen05-20261018-r1/gate.json`.

## Backbone and pilot

Qwen/Qwen2.5-0.5B base, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, uses an existing supported decoder
and cached pinned weights, and permits a relatively inexpensive K80 comparison
with earlier Qwen LSTM work. This is a new experiment, not a matched comparison
against previously trained LSTM checkpoints. Pretrained files, document corpus,
offsets and frozen source are checked against SHA256 identities.

Two arms: block summaries with null sink, and trained null sink with memory
reads disabled. Both freeze every pretrained parameter. Train on the same
ordered whole WikiText articles, reset at authoritative boundaries, seed1729,
128 optimizer updates, at most65,536 input predictions, AdamW learning rate0.01,
window/block128, two256-token segments per connected update. Forward history
continues to the full article length, but gradients detach every512 tokens.
The null arm trains only null score weight/bias. The block arm also trains the
three role biases and learned pooling query. Fixed final checkpoint; no
validation/test checkpoint selection. Save training state every16 updates;
this initial runner does not implement automatic resubmission/resume.

Evaluate the longest validation article and two longest test articles, selected
by length only before losses, each in full (up to approximately16K tokens).
Report token-weighted CE overall, after first publication, and in 0–4K,
4–8K, 8–12K, 12–16K ranges. Store per-token losses and mean null/role attention.
Test the block checkpoint enabled, with reads disabled, with values zeroed
while retaining logits, each role alone, and pure FIFO (all virtual reads off).
Run trained null and pure FIFO on the null checkpoint. Every intervention
starts a separate full-article trajectory, so effects include changed cache
activations and importance weights. Controls match physical capacity, not total
state or trainable-parameter counts. The pure FIFO trajectories should agree
across arms because the pretrained backbone is frozen.

This is a one-seed viability screen with a small training budget and only two
long test articles. Near-zero content effects may reflect inadequate training
or suppressed reads; they do not establish that block compression cannot work.

## Submission

Isolated release: `artifacts/block-memory-qwen05-20261018-r1` locally and
`/cluster/users/alex116u1/data/block-memory-qwen05-20261018-r1` remotely.
Source archive and manifests preserve the submitted bytes without uploading
credentials or overwriting other experiments. The runtime shell is included
and hashed. All batch scripts source `hpc/hpc_runtime.sh` inside Slurm.

- Gate:217401, 45-minute GPU limit, comp002.
- Block pilot:217399, 12-hour GPU limit, afterok217401, comp002.
- Null pilot:217400, 12-hour GPU limit, afterok217401, comp002.

Original gate217398 was allocated on comp014 but its batch shell stalled
reading the shared filesystem, with no stdout/stderr after more than90 seconds.
A direct hostname step responded but reading the release script stalled.
Replacement217401 uses the identical frozen source on comp002, which completed
the previous Qwen reset evaluation. Pilot dependencies were changed before
cancelling217398, and both pilots were constrained to comp002. The replacement
is recorded in `startup-replacement.json`.

Submission alone does not confirm execution. Collect logs and accounting with
`python hpc/collect_block_memory.py`; inspect at least20 seconds after submission.
If the gate fails, dependent pilots are killed rather than run on unchecked code.

Execution verified: replacement gate passed, exit0. Block pilot217399 started
on comp002 and completed its first512-token update (CE5.07716, finite gradient
norm1.69665,78.70 seconds). Null pilot217400 cleared its dependency and remains
queued for node resources/priority. Stdout and stderr were inspected more than
20 seconds after submission; block stderr was empty. Long-article quality
results are pending. Training CE from the first update is not held-out evidence.
