# Qwen2.5-0.5B full-block teacher pilot

Use pretrained Qwen2.5-0.5B Base, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, rather than random initialization.
Uptraining directly asks whether recurrent memory can be added to an existing
language model. From-scratch training would confound memory learning with
learning language and needs a substantially different budget. A later
architecture study should compare both initializations at declared, suitable
compute/data budgets; this pilot cannot answer the eventual architectural ceiling.

Release: `artifacts/qwen05-full-block-teacher-20261014-r1`; remote root
`/cluster/users/alex116u1/data/qwen05-full-block-teacher-20261014-r1`.
Preparation 217093; CUDA gate 217094; two-arm array 217095; report 217096.
All use the Bibalex runtime; GPU jobs exclude comp013 and comp016.

This is a short **language-model feasibility screen**, rather than another
random-model recall experiment. Existing pinned model and original WikiText-103
parquet assets are reused and checked against hashes. Each independent example
is the first 129 tokens of an official-split row of sufficient length. State
resets at every example, and identical token excerpts are excluded across splits.
The preparation saves source file hashes and exact row indices: 64 training
examples, eight validation examples and eight test examples. These are row
excerpts, not complete articles or long-document evaluation. No token stream
is presented as having document boundaries that its manifest cannot establish.

The memory cache holds 32 tokens; model inputs contain 128 tokens and adjacent
next-token targets. Three LSTM memory slots plus one null sink are attached to
every layer/head. Retention initialization uses 128/512/2048 eviction writes,
and initial virtual attention mass is calibrated to 0.0001 on the first training
example. Calibration does not establish a bound on unseen contexts. No explicit
read-gate objective is used, following its unstable content-learning outcome in
the toy factorial.

Both arms receive ordinary next-token CE on all 128 positions and a second,
equally weighted CE over the 96 post-eviction positions. The control's second
forward is ordinary attention. The treatment's second forward uses memory-only
attention after eviction at every layer, passing through the actual O projection,
query residual, MLP residual, final norm and LM head. Before eviction the teacher
uses ordinary attention. Each teacher forward computes its own complete causal
trajectory and memory states: it never reconstructs a multilayer teacher from
ordinary-forward states. Both arms execute two forward/backward passes per
example with matched target counts. Ordinary inference receives no forcing.

Seed 1729; 64 steps; 8192 unique input tokens per arm. Warmup freezes the backbone
for 16 steps and trains memory parameters only. The remaining 48 steps update
the entire active backbone at LR 0.00002 and memory at 0.0003. AdamW memory
moments persist across the transfer, backbone equality is asserted after
warmup, and no recalibration removes learned reads. Whole-layer checkpointing
and chunked vocabulary projection bound memory use on K80; separate backward
passes accumulate the two objective gradients before one clipped optimizer step.

Validation/test record ordinary CE, post-eviction CE, memory attention mass,
and evaluations with writes, reads, values or retention removed. Forced-teacher
CE is a separate diagnostic. The memory-disabled FIFO evaluation uses the same
current backbone; after adaptation it is not the original pretrained model.
Initial ordinary/memory-disabled evaluations provide the pretrained reference.
The frozen release also checks an initial hidden-state drift tolerance of 0.05
on its calibration example.

Before the array, a CUDA smoke job runs unit tests and two actual teacher-arm
optimizer steps: one frozen-backbone and one full-backbone step. It records
peak reserved GPU memory and timing. Array jobs request at most four hours each.
Sources are archived and hashed before submission. Read-only collection uses
`python hpc/collect_qwen05_teacher.py`.

One seed, eight held-out excerpts per split, a small cache and only 8192 training
tokens make this a feasibility and regression screen. Teacher loss improvement
alone is insufficient: ordinary held-out CE and dependence on memory must be
examined. There is no matched trained memory-free control, from-scratch run,
general recall benchmark, or long-context quality conclusion in this pilot.

## Launch verification, 14 October 2026

Preparation completed successfully. The CUDA gate passed 20 unit tests and
both real-model optimizer steps, including full-backbone AdamW. Peak reserved
GPU memory was 9,464,446,976 bytes (about 9.46 GB). The two training arms started
at approximately 09:45 Cairo. Model and corpus identities are in `identity.json`;
all 58 frozen source files passed archive/executed-source SHA256 checks in
`source-audit.json`. Results remain pending at launch.

The smoke test's first held-out excerpt had initial CE 6.05352 with memory
enabled versus 5.86107 through memory-disabled FIFO. The calibration-example
maximum hidden-state drift was 0.01283. Thus a small calibrated mass on one
training excerpt did not guarantee unchanged held-out loss. These are explicit
initialization diagnostics, not training gains; the full eight-excerpt initial
validation reference is saved separately for each arm before training.
