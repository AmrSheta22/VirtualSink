# Larger-context SWA comparison

User selected context length 1,024 and a 512-token sliding window. Earlier pilots
used 256-token full causal attention; they did not use SWA. The new window
includes the current token and the preceding 511 physical tokens, with no
special BOS retention. RoPE uses unchanged positions within the full 1,024-token
sequence. The virtual sink remains available at every query.

Four independent one-GPU jobs are submitted without serial dependencies between
them, allowing both base controls and sink runs to execute concurrently where
Slurm capacity permits. Concurrent starts are not guaranteed by queue submission.

| Model | Update scope | Variants |
|---|---|---|
| SmolLM-135M | Full backbone | Query virtual sinks; standard attention |
| Qwen2.5-0.5B | Attention projections only | Query virtual sinks; standard attention |

All four start from the original pinned pretrained weights, not the previous
uptrained checkpoints. This gives matched starting weights for each comparison.
Standard controls retain the same SWA mask but disable/freeze the added virtual
sinks and use no auxiliary penalty. They still permit ordinary physical attention
sinks among tokens present in the window. Each pair shares token streams, seed,
ordering, learning rates, schedule, update scope and evaluation blocks.

Each job trains for 1,000 steps with microbatch 1 and accumulation 8, consuming
8,192,000 input tokens. This preserves the earlier token budget while increasing
sequence length fourfold; it changes the effective number of sequences per step
from 32 to 8. Each sequence predicts 1,023 targets. Data comes from the already
prepared, pinned WikiText streams without redownload or repetition. Each training
evaluation uses 32 sequences, or 32,736 prediction targets. These metrics are not
directly comparable to the older 256-token evaluation protocol.

Checkpoint cadence remains every ten optimizer steps or ten minutes, at step
boundaries, with last/previous, best and 250-step milestones. CPU/GPU memory,
step times, checkpoint durations, validation CE and per-head physical gates are
logged. Jobs reserve 24 hours and checkpoint/pause at 23.5 hours. Longer contexts
increase attention arithmetic even though initial memory measurements were low.
The SWA implementation is an exact dense masked reference, not a sparse/fused
kernel; the window does not by itself make score computation linear. Measured
ETA updates begin after five steps and include saving/evaluation overhead.

A separate evaluation job depends on successful completion of old Qwen job
214320. It loads its best checkpoint and compares frozen trained virtual sinks
enabled, disabled, and bias-only (query weights zeroed), plus the original model
under the same context/window. It evaluates up to 256 test blocks, saving every
block's CE and paired disabled-minus-enabled CE. Positive differences mean
enabled sinks improve predictions; negative differences mean they hurt.

Additional comparison jobs depend on successful completion of both new arms for
each model. They repeat the interventions at context 1,024/window 512 and also
evaluate the matched standard-attention uptrained checkpoint. At most 256 full
blocks are used, limited by the prepared test stream (about 140 blocks). All
conditions within an evaluation use identical blocks. Bootstrap intervals
resample blocks; adjacent blocks are correlated and they are not seed-level or
independent-document uncertainty estimates. Evaluation does not update weights.

The static intervention preserves trained scalar sink biases but removes learned
query weights. Its effective gate still depends on the physical attention
partition. Original and trained-checkpoint hashes, selected steps, training
configuration, per-block CE, head gates and a summary are written to fresh
evaluation directories. A sink-disabled intervention measures sensitivity of
the trained model, not whether it would have trained equally well without sinks;
the matched standard controls address the latter question.

No recurring monitor is enabled. Releases are committed and transferred directly
to the authorized HPC account as Git bundles, with isolated checkouts; no GitHub
publication is attempted and previous running jobs/checkpoints remain untouched.
