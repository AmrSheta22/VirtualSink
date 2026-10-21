# Recurrent-memory initialization study on Bibalex — 13 October 2026

This is the next controlled experiment after the K80 Qwen implementation gate
and the two exploratory toy recall pilots. The earlier success changed write
initialization, read initialization and training duration together. This study
separates those factors and tests whether the conservative initialization used
for pretrained insertion can learn with a delay curriculum.

## Frozen protocol

Remote release: `/cluster/users/alex116u1/data/recurrent-init-factorial-20261013-r1`.
Local provenance: `artifacts/recurrent-init-factorial-20261013-r1/`.
The source archive and plan were frozen before submission; running experiment
sources, previous releases and unrelated jobs are not edited.

Three paired seeds: **1729, 1730, 1731**. Eight arms per seed (24 tasks):

| Arm | Maximum observed initial total virtual mass | Input-gate bias | Training delay |
|---|---:|---:|---|
| low_weak_fixed | 0.0001 | -4 | 24 distractors |
| low_strong_fixed | 0.0001 | -1 | 24 distractors |
| high_weak_fixed | 0.5 | -4 | 24 distractors |
| high_strong_fixed | 0.5 | -1 | 24 distractors |
| low_weak_curriculum | 0.0001 | -4 | 8 → 24 over the first 300 updates; then 24 |
| low_strong_curriculum | 0.0001 | -1 | Same curriculum |
| standard_fixed | 0 | Unused | 24 distractors |
| standard_curriculum | 0 | Unused | Same curriculum |

The read-mass factor changes the common virtual-score bias, preserving the
initial null/memory logit relationship. Both write settings use nonzero random
write/recurrent/read transforms and zero initial stream state. The low-mass arms
use the production calibrator with default maximum-observed-mass targeting.
The high-mass arms first use that same calibrator, then apply a toy-only common
logit shift to achieve 0.5 maximum observed mass. The one-layer model's prospective
QKV/write inputs are independent of attention outputs, making this shift exact
on its calibration examples. This shortcut is not used for multilayer Qwen.

For each seed, the base model and transformation matrices are identical across
arms. Fixed-delay arms share the exact training example order; curriculum arms
share their own exact order. Changing delay changes how many random distractors
the generator consumes, so target order also differs between fixed and curriculum
training. A separate calibration generator avoids consuming training examples.
Initialization reports measured mass and maximum logit deviation from standard
attention. High-mass arms are optimization diagnostics, not a recommended
pretrained initialization.

## Task and optimization

Random one-layer native Qwen-style decoder, vocabulary 32, hidden dimension 32,
four query heads/two KV heads, head dimension eight, MLP dimension 64, tied word
embeddings. Three LSTM slots and one null sink per query head; FIFO capacity
eight; retention timescales 128/512/2048 eviction writes. There are 1,328 new
memory parameters. All active base and memory parameters are trained from
scratch. Standard controls disable writes/null/memory and freeze memory parameters.

Each example has a target class token in 1–8 at the start, independently random
distractors from 9–31, then query token zero. Only the final query is supervised.
Its target is the first token. Even the first curriculum query occurs after
fact eviction. A one-layer physical cache cannot relay contextual fact content:
its K/V come directly from token embeddings, and its remaining suffix is
independent of the target. Chance accuracy is 12.5%.

Each arm uses 1,000 updates, batch 32, fixed AdamW learning rate 0.003, base weight
decay 0.01, memory decay zero, and gradient clipping at one. No auxiliary loss,
early stopping, or held-out model selection. Report fixed snapshots at **0, 300,
1,000** updates along the same trajectory; step-300 evaluation does not consume
the training generator. Fixed-delay arms have 32,000 supervised labels and
832,000 input tokens at completion. Curriculum arms have the same label/update
budget but fewer input tokens; record that explicitly rather than claiming a
matched input-token budget.

Held-out delays: 24, 48, 96, with the same 512 examples at each delay for every
seed/arm/snapshot. Record accuracy, CE, correct-label probability, and final-query
null/memory mass. Memory-arm interventions disable writes, disable reads, zero
memory values while retaining scores, or force the forget gate to zero. These
are inference interventions on the same checkpoint, not independently trained
ablations. They establish dependence on the mechanism rather than a fair
performance comparison by themselves.

## Preregistered comparisons and limits

Primary: held-out delay-24 accuracy and CE at step 1,000. Pair seeds/examples for
read-mass and write-bias contrasts among fixed-delay arms. Step 300 versus 1,000
measures the budget effect within each trajectory. Curriculum comparisons
change training delay distribution, input-token count and generated example
order, holding label/update budget fixed. Secondary: longer-delay generalization, query mass, and intervention
sensitivity. Preserve all results, including failed-to-learn seeds.

Report three-seed means, ranges, individual outcomes and paired differences;
do not present tight confidence intervals from three training seeds. Completing
this task is not evidence of a pretrained Qwen language-model improvement.
Learning under low initial mass would justify a subsequent pretrained-memory
warmup experiment. If it fails, diagnose initialization/gradient access before
committing to expensive corpus uptraining.

## Execution and collection

- Validation **216955** runs the 22 numerical/state/training/calibration tests.
- Training array **216956** has tasks 0–23, concurrency six, depends on successful
  validation, and cancels if the validation dependency fails.
- Report **216957** runs after array termination. It requires every task's
  completion marker and all three snapshots/policies with verified identities
  and label/evaluation counts. Missing runs produce `INCOMPLETE.json` and no
  complete aggregate report.

Every batch script sources the HPC runtime; GPU jobs exclude comp013/comp016.
Each task has a 30-minute allocation. Runtime and completion estimates must use
observed task progress; the earlier 162-second memory toy run is only a planning
reference. The new array calibrates and evaluates more checkpoints.

```bash
PYTHONPATH=src python hpc/deploy_recall_experiment.py
python hpc/collect_recall_experiment.py
```

Deployment refuses existing local/remote release roots. The collection script
only reads Slurm state and copies small logs/JSON reports; final model and
optimizer checkpoints remain in the frozen remote run directories.

## Verified outcome

All 24 tasks completed. Report 216957 finished at **11:18:40 Cairo** on 13 October.
The campaign including validation/reporting took 11m20s; the array ran for 9m40s.
The executed source hashes, 72 snapshots, input/label budgets and aggregate
accuracy/CE means were independently checked. Results are in
`artifacts/recurrent-init-factorial-20261013-r1/FINDINGS.md` and `statistics.json`.

At step 1,000, low-mass fixed arms achieved 12.0% mean delay-24 accuracy, matching
standard; low-mass curriculum arms achieved 13.6%, matching their standard.
High-mass/weak-write achieved 67.3% (individual seeds 11.3%, 90.4%, 100%).
High-mass/strong-write achieved 100% in all seeds, already reached at step 300.
Its write/read/value/forget interventions reduced recall to chance.

Long-delay performance still deteriorates: the strong-write arm averaged 85.4%
at delay 48 and 50.7% at delay 96, with delay-96 CE worse than standard. Weak-write
successful seeds generalized farther but one seed failed to learn. A direct
memory-learning objective under conservative read mass is the next proposed
bootstrap test before pretrained uptraining; these findings do not establish
Qwen streaming gains.
