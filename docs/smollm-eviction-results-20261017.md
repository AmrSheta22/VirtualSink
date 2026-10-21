# SmolLM prefix-replacement screen: completed results

Collected17 October2026. Preparation217231, all six GPU validation units217232,
all30 evaluation units217233 and report217234 completed with exit0. All360
policy outputs are present:1,474,560 scored next-token predictions. Frozen
identities and finite per-token losses passed the aggregator's checks.

## Main finding

This screen does not establish replacement of StreamingLLM's retained initial
tokens. Retaining the first four tokens barely helps even the standard
checkpoints. Thus these short-trained models do not exhibit a substantial
prefix-retention rescue on the tested streams. Similar FIFO/prefix results
with a virtual sink cannot establish that the sink caused independence.

The table gives post-eviction CE, lower is better, averaging equally across
five fixed held-out source streams, input positions512–4095. All columns here
use cache-relative RoPE. Absolute-position controls agree that prefix effects
are small; the complete per-source and policy results remain in aggregate.json.

| Seed | Standard FIFO | Standard prefix4 | Virtual FIFO, enabled | Virtual FIFO, disabled |
|---|---:|---:|---:|---:|
| 8 | 7.616640 | 7.616640 | 7.312290 | 7.312219 |
| 9 | 7.613460 | 7.613460 | 7.619258 | 7.619739 |
| 10 | 7.612708 | 7.612704 | 7.611458 | 7.610895 |
| 11 | 7.053297 | 7.051979 | 6.551118 | 6.551354 |
| 12 | 6.680996 | 6.681290 | 6.594857 | 6.597772 |
| 13 | 6.545251 | 6.545809 | 6.614395 | 6.615921 |

Seeds8–10 trained30 updates; seeds11–13 trained70 with10-update warmup.
Do not combine those cohorts into an estimated training-seed effect.

## Prefix-retention and direct sink effects

Standard FIFO CE minus prefix4 CE under cache-relative positions, positive
meaning prefix retention helps: seed8 −0.00000002; seed9 −0.00000061;
seed10 +0.00000425; seed11 +0.00131724; seed12 −0.00029337;
seed13 −0.00055845. The largest average rescue is seed11, equivalent to
approximately0.132% lower perplexity. The largest per-source relative rescue
is seed11/Python at0.002709 CE. No large rescue is hidden by source averaging.
Under absolute positions, standard prefix rescues range from−0.00217444 to
+0.00052906 CE.

Enabling the sink within its own trained checkpoint, under relative FIFO,
reduces perplexity by approximately−0.0071%,+0.0480%,−0.0562%,+0.0236%,
+0.2911%,+0.1525% for seeds8–13 respectively. Negative means a small worsening.
These direct effects are much smaller than the cross-checkpoint advantages
of seeds8/11/12. The main gains continue to survive disabling the sink,
consistent with a training-path effect rather than demonstrated replacement.
The present evaluation does not establish the precise causal training mechanism.

## Interpretation and limits

The original question remains unresolved for a model that actually depends on
retained initial tokens. The missing positive control is a substantial
StreamingLLM-style rescue without a virtual sink. Short training is a plausible
reason that dependence is weak here, but this screen does not prove that cause.

Use a checkpoint with a demonstrated FIFO/prefix-retention gap on the same
long-stream evaluation before measuring whether a virtual sink closes it.
The proposed eviction-aware training or learned-value changes remain untested;
no further training was launched for this screen.

Five EOS-packed source streams are a descriptive screen, not independent
long-book replication. Post/pre-eviction means cover different target tokens
and cannot by themselves measure eviction damage. Cache-relative decoding also
changes the query-conditioned virtual-sink score, so absolute and relative
controls are interpreted separately. Tiny differences do not prove statistical
equivalence, and no confidence claim is made from correlated token samples.

Artifacts: artifacts/smollm-eviction-20261016/aggregate.json, identity.json,
protocol.json, validation/seed8.json through validation/seed13.json.
