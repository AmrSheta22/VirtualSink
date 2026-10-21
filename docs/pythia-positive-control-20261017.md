# Prefix replacement on checkpoints with a measured streaming gap

Authorized17 October2026. This follows the inconclusive short-trained SmolLM
prefix-replacement screen. Use existing immutable Pythia410M checkpoints and
the pinned PG19 token stream; no training or changes to previous jobs.

The completed400k-prediction standard-uptrained checkpoint gives FIFO-absolute
perplexity122.388 versus20.168 under prefix4/cache-relative StreamingLLM.
The original pretrained Pythia2.8B and Qwen1.5B also show large gaps, but using
the already trained410M pair provides a direct test without new uptraining.
Legacy and calibrated trained sinks have FIFO/StreamingLLM perplexities
319.586/24.890 and120.733/20.217. These completed results already show that the
tested enabled sinks did not eliminate that bundled retention/position gap.

On the exact first65536 predictions, post-eviction CE gaps are1.805753 for
standard,2.634967 for legacy, and1.799806 for calibrated. These are not pure
prefix effects: cache-position policy changes too. The new factorial controls
separate retention from position and isolate the sink's direct contribution.

## Fixed design

Four arms: untouched pretrained Pythia410M; standard final256-step uptraining;
legacy and calibrated zero-value-sink final256-step uptraining. All FP32, same
immutable pretrained base/revision and PG19 tokens. Old best checkpoints are
not substituted for the matched final checkpoints.

For each arm cross prefix0/4 with absolute/cache-relative RoPE. For each sink
arm also cross enabled/disabled in the exact same checkpoint. Total24 policies,
65536 predictions each,1,572,864 scored targets. Six evaluation tasks may run
concurrently. Cache budget1024 stored entries;1025 transient attention entries,
matching the original released evaluator's post-prediction trim convention.
Do not reset cache at book boundaries, inject EOS, recompute historical hidden
states, or score a cross-book target. This is the same previously inspected
test stream, not an independent new-document confirmation.

Primary region1024:65536; secondary first1024 and10240:65536. Preserve full
per-token losses and book-boundary identities. Report within-checkpoint prefix
rescue with sink enabled/disabled, FIFO sink-enabling benefit, and enabled FIFO
versus standard retained-prefix performance separately for each position policy.
No confidence claim from one training seed or correlated token samples.

Successful replacement requires a substantial prefix benefit while sinks are
disabled and reduced dependence when enabled, with competitive enabled FIFO
performance. Absence of prefix benefit even when disabled is uninformative.
The untouched pretrained arm checks whether standard uptraining changes the
failure mode. Full512-block uptraining did not explicitly train for eviction;
this test cannot establish the effect of eviction-aware training or learned values.

## Gates and execution

Remote root `/cluster/users/alex116u1/data/pythia-positive-control-20261017`.
Local provenance `artifacts/pythia-positive-control-20261017`.
Preparation217334 verifies old checkpoints against their saved hashes, pins
model safetensors/config/tokenizer/data hashes, and requires the original
same-budget FIFO-absolute/prefix4-relative CE gap to exceed0.2 for every trained arm.

GPU validation217335 has four units, one per arm, and tests all24 policy
configurations against the official HF cache path over1153 tokens including
eviction. Absolute paths use the stock HF cache; relative paths use pinned
official raw-key shifting. Tolerances0.003 absolute/0.0001 relative. Frozen
attention and decoder files are copied verbatim from the completed old release.
The evolving workspace attention implementation is not used.

Evaluation217336 has24 independent tasks, concurrency6, and starts only after
all validation units succeed. Six configurations matching old trained policies
must reproduce their original first65536 per-token losses within0.003. Each
evaluation has a six-hour allocation; validation has two hours. No automatic
retry or retraining. Report217337 runs after evaluation termination and writes
aggregate.json only with all24 finite loss arrays and matching identities and
book boundaries; otherwise it writes INCOMPLETE.json.

Local statistical/design checks: four tests passed. These separate position
from retention in the contrasts and reject missing or corrupted results. Real
model decoder validation runs on Slurm before evaluation.

Collect using `venv/Scripts/python.exe hpc/report_pythia_positive_control.py`.

## Node recovery, 17 October afternoon

Preparation passed. Pretrained, standard and calibrated validation completed in
11m36s,12m00s and23m36s, respectively. The legacy validation lost comp015 to an
unexpected node reboot, was automatically restarted by the cluster on comp014,
and then timed out at2h00m21s without producing a validation result. comp014
remained in COMPLETING for hours. Original evaluation217336 never started.
The earlier14:30–16:00 Cairo estimate is withdrawn.

Recovered manually with hpc/recover_pythia_positive_control.py: verify all three
successful validation files, preserve jobs-before-node-recovery.tsv, cancel
only pending217336/217337, and submit legacy-only validation217355 on comp002.
New evaluation217356 (24 units, concurrency6) depends on successful recovery;
report217357 follows evaluation termination. Source, model/data identities,
policies, budgets and tolerances are unchanged. Recovered GPU tasks disable
automatic requeue and evaluation additionally excludes comp004/005/014/015,
which were completing/down when inspected, alongside original comp013/016.
The old legacy job217335_2 remains recorded as a timeout/stuck COMPLETING;
it has no outputs to reuse. No training or unrelated jobs were modified.
