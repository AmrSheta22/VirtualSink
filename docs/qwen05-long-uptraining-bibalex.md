# Qwen0.5B longer-context uptraining

On 16 October 2026 ordinary-memory training migrated safely from legacy job
217102 to optimized job 217217 (backup 217218), resuming fully verified
checkpoint 1157 with optimizer and both stream memories intact. It produced
four finite updates through 1161 at about 31.8 seconds/update, versus roughly
158 seconds before migration. Six-hour monitoring now follows the new jobs.
The first handoff helper encountered a corrected Slurm memory request error;
training state was preserved. See
[the migration audit](qwen05-optimized-handoff-bibalex.md) for provenance,
checkpoint verification, the rollback checkpoint and job/report history.

Frozen release `artifacts/qwen05-long-uptraining-20261014-r1`, remote
`/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1`.
This continues the full-block-teacher investigation with a larger budget,
starting each arm from the original pretrained Qwen2.5-0.5B Base checkpoint,
revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

## Training budget and context

- Memory arms: 2048 optimizer updates, versus 64 in the feasibility pilot.
- Each update processes two contiguous 256-token segments, up to 512 distinct
  input/target pairs. Article-ending segments can be shorter. Budget is roughly
  one million input tokens per arm, at most 1,048,576. No corpus repetition.
- Physical cache 128 tokens, versus 32 previously. Recurrent state persists
  through complete articles of 1025–16384 tokens, resetting at article boundaries.
- Gradients span the two connected segments (512 tokens), then both physical
  cache and recurrent state detach. This is TBPTT, not a 16K-token gradient graph.
  State carried after optimizer updates was computed under previous weights.
- Three retention initializations: 512, 4096 and 16384 eviction writes.
- One seed (1729). Global memory LR warms for 64 steps then follows cosine decay
  to 10% of peak 0.0003. Backbone is frozen for 128 steps, then all active backbone
  parameters update with a 64-step LR warmup and cosine decay from peak 0.00001.
- Initial virtual mass is calibrated to 0.0001 on four distinct training articles,
  using 512 tokens each. This is an observed initialization target, not a bound
  on unseen documents. No later recalibration erases learned reads.

## Controls and teacher

Three parallel arms: trained FIFO without virtual attention, ordinary recurrent
memory uptraining, and recurrent memory with the full-block teacher. The FIFO
control performs no parameter updates during the matched 128-step memory-only
warmup, then receives the same 1920 backbone updates as the memory arms.

All arms perform two causal passes over the same data per update, with equal
next-token CE weights over all valid targets. The second teacher pass forces
memory-only attention after eviction; its own hidden states, K/V cache and LSTM
states evolve through the actual residual/MLP blocks. Before eviction it uses
ordinary attention. Other arms use ordinary attention in both passes. This
keeps the target and pass counts matched. It differs from the pilot's extra
post-eviction-only loss. No explicit read-gate BCE is added.

Ordinary and auxiliary trajectories have separate persistent states. They
start each update at identical data cursors; their counts/end cursors are
asserted equal. Each trajectory resets independently at the same article
boundary. Teacher forcing never occurs during ordinary evaluation.

## Data

Pinned original WikiText-103 official-split parquet assets and the pinned Qwen
tokenizer are hash checked. Top-level title rows define complete articles;
subsection headings do not reset state. Unclosed shard-edge articles are
discarded, as are articles outside the token-length limits. EOS is appended
at each genuine article end. Explicit token offsets are authoritative; no
article boundaries are inferred from EOS occurrences inside text. Exact
token-identical articles are deduplicated across selected splits.

Preparation targets two million training tokens and requires at least 1.1M.
That exceeds the maximum input budget, including article tails. Source hashes,
row ranges, article lengths, token hashes and offset hashes are retained. Final
validation/test each use four separate articles longer than 4096 tokens, scoring
their first 4096 predictions. Intermediate validation uses the first eligible
validation article for 2048 predictions every 256 updates. Best checkpoint
selection uses post-eviction validation CE; test is read only after training.

## Checkpoints and evaluation

Every 64 updates an atomic checkpoint saves model, Adam moments, both stream
states, data cursor, token/step counts, learning-rate phase and RNG state.
Resume identity includes source and corpus hashes and all training settings.
Backbone equality is checked at warmup transfer. Best-validation weights are
saved separately, limiting storage to the latest resumable checkpoint and
best weights per arm. Intermediate validation precedes checkpoint saving so
best-selection metadata stays consistent after resume.

Final evaluation reports CE/PPL overall, after eviction and beyond ten physical
windows (1280 tokens), plus average null/memory mass. Memory arms are evaluated
with reads, writes, values or retention removed, alongside a fully disabled
virtual path. The trained FIFO control distinguishes backbone adaptation from
an actual benefit of recurrent memory. One seed and the held-out token budget
still limit broader claims.

## Slurm execution

Preparation 217098; full-size GPU gate 217099. Primary jobs FIFO 217100,
ordinary memory 217102, full-block teacher 217104. Continuations 217101,
217103 and 217105 depend on their respective primary jobs; report 217106 waits
for continuations. Each allocation requests one K80, four CPUs and 24 GB host
memory, excludes comp013/comp016, and allows seven days. The process pauses
after 160 hours or a termination signal and saves a checkpoint; the continuation
loads it automatically. If already complete, continuation exits immediately.
Two allocations bound the automatic schedule; the report fails explicitly
if the requested training/evaluation is unfinished.

The GPU gate executes unit tests, a full 512-token teacher update with frozen
backbone, another full-backbone update at the next stream position, and an
actual checkpoint save/load. Production cannot start until this passes. Local
verification passed 22 tests, including exact next-update equivalence after
resuming both stream states and the optimizer. Earlier immutable experiment
releases remain unchanged.

Expected duration is several days on K80; use the full-size profile and actual
training speed for estimates. Read-only collection:
`python hpc/collect_qwen05_long.py`. Checkpoints remain on the cluster.

## Launch measurements

Preparation produced 2,002,450 training tokens in 484 articles, and 57 articles
each in validation/test (234,553/293,351 tokens). All 598 article identities are
distinct. Deterministic cursor simulation gives 1,016,397 input tokens per arm
for 2048 steps, completing 241 articles plus 768 tokens of the next. The longest
context reached is 14,995 predictions; the prepared corpus's longest article
is 15,891 tokens. See `expected-budget.json` and `data-audit.json`.

The frozen cluster gate passed 22 unit tests. An additional local full-backbone
resume case also passed after source freezing, supplementing the memory-only
resume test. Both real-model profile updates finished with finite gradients:
142.50 seconds for the frozen-backbone update and 158.05 seconds for the
full-backbone update. Peak reserved memory reached 9,690,939,392 bytes. An
extrapolation from these two updates suggests roughly four days for a memory
arm including evaluation; it is provisional until steady production throughput
is measured. `source-audit.json` verifies all 60 archived/executed source files.
