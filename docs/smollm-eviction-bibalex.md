# SmolLM prefix replacement screen, 16 October 2026

Evaluate the completed standard/query-sink checkpoint pairs for seeds8–13.
Seeds8–10 trained for30 updates; seeds11–13 for70 updates with warmup10.
Keep those cohorts separate. This evaluation does not train or alter checkpoints.

Remote root: `/cluster/users/alex116u1/data/smollm-eviction-20261016`.
Local protocol and frozen evaluator hashes: `artifacts/smollm-eviction-20261016`.
Remote `jobs.tsv` is authoritative. Preparation217231 pins checkpoint hashes,
paired configs/shared initialization hashes, numerical source and original test
file hashes. All numerical backends come from the original frozen training
releases, rather than the workspace's evolving multi-sink implementation.

GPU validation array217232, six units, must complete before evaluation217233
can start. Validation runs the local five-test suite again in the K80 runtime
and compares both checkpoint arms' enabled/disabled cached logits with their
parallel sliding-window logits over641 tokens, including129 post-eviction
positions. A separate absolute-key/delta-rotation oracle validates rebased
prefix retention. Maximum logit tolerance is0.003 absolute/0.0001 relative.

Evaluation array217233 has30 units (six seeds by five sources), concurrency6.
Each scores the first4096 next-token predictions from the original held-out
FineWeb, Cosmopedia, Python, math and StackOverflow streams. Selected token
and full stream hashes must match across both training experiments. Every
policy within a unit reads identical token IDs, starts with an empty cache,
and retains historical hidden-state K/V without recomputation or resets.
The streams contain EOS-separated documents. This is a controlled packed-stream
screen, not a long-book replication or a new independent held-out dataset.

There are twelve policies per unit: standard checkpoint with FIFO/prefix-four;
virtual checkpoint with FIFO/prefix-four and sink enabled/disabled; all six
under both absolute and cache-relative RoPE. Physical cache capacity512
includes the current token; trim occurs before attention. Prefix retention
keeps four initial and508 recent entries. The virtual sink consumes no physical
entry and has the original zero value. Relative decoding stores unrotated keys
and rotates retained entries at current cache indices. This also changes the
query used to score the virtual sink; interpret absolute and relative results
separately. This is an explicit positional control, not exact replication of
every upstream StreamingLLM cache convention.

Primary region: input positions512–4095. Secondary: pre-eviction0–511 and
late2048–4095. Preserve raw per-token CE and mean sink mass. Total scored
predictions:1,474,560. Report per source and per seed, with equal source weights
for a descriptive summary. Do not interpret the five source streams as many
independent documents, or pool the different training budgets into one estimate.

The key comparisons are within the virtual checkpoint: prefix rescue with sink
disabled, prefix rescue with sink enabled, and the benefit of enabling the sink
under FIFO. Successful replacement requires evidence that prefix retention
helps without the sink, that this dependence decreases when enabled, and that
enabled FIFO approaches enabled prefix performance. Cross-checkpoint CE alone
cannot establish replacement. If the standard checkpoint has no prefix rescue,
the canonical StreamingLLM failure mode was not demonstrated for that seed.
No small or nonsignificant difference establishes equivalence automatically.

CPU report217234 runs after the evaluation array terminates and writes
`aggregate.json` only when all30 units and360 policy files are present and
finite with matching identities. Otherwise it writes `INCOMPLETE.json`.
Individual GPU tasks have12-hour allocations; there are no automatic retries.
Existing training and unrelated evaluation jobs are untouched.

Collect status and completed summaries with:

```powershell
venv/Scripts/python.exe hpc/report_smollm_eviction.py
```

Local validation: five decoder/design tests passed; two additional summary tests
passed, rejecting missing units and corrupted provenance. The frozen cluster
release repeats the original five decoder/design tests before real-checkpoint
validation. Summary tests were added locally after submission; evaluator source
is unchanged.

The first preparation217227 failed at shell startup because the reused runtime
required WARM70_ROOT to be exported. No scientific computation ran. Dependent
arrays217228/217229 were automatically cancelled and report217230 failed at the
same startup line. Preserve jobs-first-attempt.tsv. Replacement launchers export
the original warm70 root before sourcing the runtime; numerical source is unchanged.
