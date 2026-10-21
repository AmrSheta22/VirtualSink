# Direct memory warmup under conservative attention — 13 October 2026

Follow-up to the three-seed initialization study: conservative initial virtual
mass 0.0001 stayed at chance, while strong read mass could learn. This experiment
tests whether an unweighted memory retrieval loss can train useful stored content
without initially giving it a large contribution to ordinary attention.

## Frozen arms and stages

Remote root `/cluster/users/alex116u1/data/memory-warmup-20261013-r1`; local
source/plan/logs/results under `artifacts/memory-warmup-20261013-r1`.
Three paired seeds 1729/1730/1731, four arms each:

| Arm | First 300 updates | Next 1,000 updates |
|---|---|---|
| normal_full | Normal recall; all active parameters | Normal recall; all active parameters |
| frozen_normal | Normal recall; only memory writes/values | Normal recall; all active parameters |
| aux_then_normal | Direct memory loss; only memory writes/values | Normal recall; all active parameters |
| aux_then_joint | Same direct memory warmup | Normal loss + direct memory loss, weight 1; all active parameters |

All arms start with measured maximum virtual mass 0.0001 and input-gate bias -4.
At update 300, every arm is recalibrated on a separate train-only sample to the
same mass before adaptation. Initial and transfer calibration are not a bound
on later learned attention mass; record the mass throughout evaluation.

The frozen warmup arms train write/recurrent matrices, gate biases, and the
memory value projection. The entire original backbone, memory key projection,
null parameters and absolute score biases are frozen. Key freezing bounds
content-score changes while values learn. Compare every original parameter
before/after warmup and require bitwise equality. The full-normal control is
allowed to change the backbone and is not expected to preserve it.

AdamW moments for previously trained parameters carry across the stage boundary;
newly enabled parameters start with fresh moments. All arms have the same 1,300
updates, 41,600 supervised labels, 1,081,600 input tokens, fixed learning rate
0.003, batch 32 and ordered training examples per seed. Base decay is 0.01,
memory decay zero, global gradient clipping one. No early stopping or model
selection based on held-out performance.

## Direct read and objective

Use the final query's unrotated projection to score the three exposed memories,
then normalize **only among those memory slots**. The unweighted memory value
mixture is passed through the existing attention output projection, final RMS
norm and language-model head. Physical attention, null sink, query residual and
MLP are excluded from this auxiliary decoder. No new classifier is added.

The auxiliary objective is full-vocabulary cross-entropy for the remembered fact.
During frozen-aux warmup it is the sole optimized objective; ordinary recall loss
is measured but does not contribute gradients. For aux_then_joint adaptation,
the two losses are summed. Auxiliary gradients may then also update the enabled
backbone. This is deliberate and differs from the frozen warmup stage.

Direct decoding removes tiny joint-attention mass from the learning path. It
does not change the ordinary attention computation. Successful auxiliary recall
alone does not show that ordinary attention can use the warmed memory; evaluate
these outcomes separately before and after transfer calibration and adaptation.

## Model/task/evaluation

Random one-layer native Qwen-style decoder, vocabulary 32, hidden size 32, four
query heads/two KV heads, head dimension eight, MLP size 64, tied embeddings.
FIFO eight, three memory slots/head, retention timescales 128/512/2048 writes.
This is a toy experiment, not uptraining a pretrained Qwen checkpoint.

The first token is the fact (class 1–8), followed by 24 independent distractors
(tokens 9–31) and query zero. Only the final query is supervised. The fact is
evicted, and a one-layer standard cache cannot relay contextual fact states.
Examples, calibration and evaluation use separate generators; all arms share
exact training example order within each seed. Unlike the previous curriculum
study, sequence lengths are identical across arms.

Evaluate at step 0, after step-300 transfer recalibration, and step 1,300. Preserve
the step-300 pre-recalibration evaluations separately in `transfer.json`.
Held-out delays 24/48/96, same 512 examples per delay across every arm/seed.
Record ordinary and direct-memory accuracy/CE separately. Ordinary interventions
disable writes, disable reads, zero values while retaining logits, or force the
forget gate to zero. Direct-memory interventions disable writes or retention.
Preserve original-backbone equality and measured original-function logit drift
on a fixed probe before/after recalibration.

Primary outcome: ordinary recall accuracy and CE at delay 24 after 1,000
adaptation updates. Compare aux_then_normal with frozen_normal to isolate the
warmup objective under identical trainable scopes. Compare aux_then_joint with
aux_then_normal to test continued auxiliary supervision. Compare normal_full
as the total-budget normal-training control. Longer delays, auxiliary recall,
mass and intervention sensitivity are secondary. Report individual seeds and
means, avoiding tight confidence claims from three seeds.

## Execution

The deployment creates an isolated source archive and source hashes before
submitting a validation gate, array 0–11 with concurrency six, and a strict CPU
report after array termination. Every script sources the Bibalex HPC runtime;
GPU allocations exclude comp013/comp016. Missing completion/transfer/snapshot
files produce INCOMPLETE.json, never a complete result.

```bash
PYTHONPATH=src python hpc/deploy_memory_warmup.py
python hpc/collect_memory_warmup.py --compact
```

The implementation tests direct gradient access, unchanged frozen backbone and
scores, optimizer moment transfer, and the absence of new decoder parameters.
Other recurrent numerical/state/training tests run in the cluster gate too.

## Verified results

Validation 217051 passed 24 tests; all 12 tasks in array 217052 completed, and
report 217053 finished at 22:15:49 Cairo. The 7m37s campaign produced 36 verified
snapshots. Auxiliary warmup preserved every frozen backbone parameter and gave
61.3% mean direct recall with probe logit drift below 0.00004 after recalibration.

After adaptation, aux_then_joint achieved 100% direct recall at delays 24/48 and
91.1% at delay 96, but ordinary attention remained near chance (14.0%/11.7%/12.6%).
Query memory mass stayed around 0.001% and ordinary predictions were insensitive
to memory ablations. Removing auxiliary supervision after warmup did not maintain
direct-decoder recall (9.2% at delay 24).

An inference-only frozen-checkpoint sweep, job 217065, raised ordinary delay-24
recall to 56.4% when memory received 99.982% of attention mass, still below direct
recall 100%. This supports investigating the normal read/merge path as well as
gate learning. Direct decoding omits residual and MLP, and cannot be taken as
successful transfer to the native attention block.

See `artifacts/memory-warmup-20261013-r1/FINDINGS.md` for verified seed-level
results, limitations, audits, and the proposed block-matched/gate-supervised test.
