# Controlled long-distance retrieval from trained Qwen checkpoints

Experiment `qwen05-long-retrieval-20261016-r1` evaluates the completed,
validation-selected step2048 checkpoints, without any further training or
checkpoint selection. Job217302 prepares the frozen dataset; GPU jobs217303
(memory),217304 (same weights, memory disabled),217305 (direct forget gate
forced to zero),217306 (trained FIFO) run on separate K80 nodes. Original
training checkpoints remain untouched; immutable hard links stay on cluster.

Each of eight item keys is assigned all eight possible single-token colors in
counterfactual copies of the same prompt. Only the stored answer token changes
within a block: key, question, background and sequence length stay identical.
The global answer set is red/blue/green/yellow/black/white/orange/pink. Every
distance has64 cases, balanced across labels; forced-choice chance is12.5%.
Four key/background blocks contain unrelated neutral prose; four additionally
contain color records for other item keys. The target key never occurs in the
distractors. The two fixed few-shot demonstrations use different keys.

Filler lengths are0,64,128,256,512,1024,2048,4096,8192,16384 tokens. The exact
distance from source answer token to answer-predicting query token is filler+16:
16,80,144,272,528,1040,2064,4112,8208,16400. Thus both key and answer are nearby
in the0-token check; at128 filler tokens the answer has left the physical
128-token cache. Maximum total sequence length is slightly above16K, so the
last point slightly exceeds the uptraining article lengths. Report true source
distance, rather than confusing absolute article position with retrieval age.

One persistent prefix scan supplies all distances. Each query branches from
the caller-owned immutable state, and its returned state is discarded, so
earlier queries/answers never enter later context. The evaluator never feeds
the correct answer after the query. Segments are256 tokens; batch32 covers
four blocks times eight color variants with aligned positions. Each arm resets
between independent batches. No gradients, optimization or calibration occur.

Controls:

- Missing-answer: replace the stored color with the single token "unknown".
  One identical stream per key is scored against all eight balanced labels;
  forced-choice accuracy must be exactly12.5% at every gap. This tests answer
  priors/leakage, rather than claiming64 independent no-answer examples.
- Repeat the fact immediately before the final query at the maximum gap.
  This checks that the model can answer locally at long absolute positions.
- Memory arm only: clear LSTM cells/exposed states immediately before the
  query while preserving physical K/V and positions. This is a selective
  state intervention; physical K/V may already encode earlier memory effects.
- Same memory-trained weights with all virtual reads/null/writes disabled.
- Direct cell forget factor forced to0 during the whole scan. Recurrent
  exposed-state gate projections remain, so this is not a strict removal of
  every theoretical recurrent information path.
- Independently trained FIFO, with all virtual reads/writes disabled.

Metrics include balanced eight-way forced-choice accuracy, unconstrained exact
next-token answer accuracy, normalized candidate NLL, absolute answer-token
NLL and counterfactual association scores. All answers are one tokenizer
token; unconstrained scoring uses the full vocabulary argmax. Confidence
intervals resample the eight key/background blocks, stratifying4+4; the eight
labels within each block are counterfactuals, not independent samples.
Intervals across ten distances are exploratory and unadjusted.

The24-layer FIFO has a finite input receptive field of24*(128-1)=3048 tokens.
Beyond that span its predictions should be identical across the eight changed
source colors: accuracy12.5%, zero counterfactual logit variation. This is an
additional finite-cache/no-leakage sanity check, not a claim that information
cannot propagate through multiple FIFO layers inside that span.

Local tests verify exact source-token differences, distance arithmetic,
balanced missing-answer scoring and non-mutating query/clear-state branches.
The same tests pass in the cluster preparation job. Small dataset, score and
log artifacts live under `artifacts/qwen05-long-retrieval-20261016-r1`;
checkpoints are never downloaded. Use `hpc/collect_qwen_long_retrieval.py`
and `hpc/analyze_qwen_long_retrieval.py` for read-only collection/local analysis.

The separately frozen exploratory follow-up
`qwen05-long-retrieval-cloze-20261016-r1` uses the identical source prefixes,
keys, labels and backgrounds, with a completion-style query: "According to the
record, item ... has the color". Its true source distances are filler+15.
It also inserts an explicit newline before the repeated near-query fact;
the original control lacked that separator. This follow-up was declared after
the original Q/A results exposed formatting/control weaknesses, and is not an
independent preregistered replication. Jobs217307 (preparation),217308 (memory),
217309 (disabled),217310 (forget=0),217311 (FIFO) use the same frozen checkpoints.
The dataset-pair audit checks that only the declared query/control changes
occurred. Matching collect/analyze scripts end in `qwen_retrieval_cloze.py`.

`hpc/report_qwen_retrieval.py` verifies frozen source hashes, completion records,
row counts and error logs before creating the combined measured report and
accuracy figure in `artifacts/qwen05-long-retrieval-20261016-r1`.
Exact unrestricted first-token color scoring is case/format sensitive and is
not a full free-form generation test. Neither protocol trains the model.

All eight GPU evaluation jobs completed with exit code zero and no restarts;
the final follow-up finished on 16 October 2026 at about 23:13 Cairo.
The completion-format memory arm scored64/64 nearby,47/64 after64 filler
tokens, and8/64 (chance) at every tested gap from128 through16384. Repeating
the fact near the final query restored57/64; the same weights with memory
disabled scored60/64 on that control. Thus usable long fact retrieval was not
demonstrated by this trained checkpoint. Tiny far-distance logit associations
do not translate into above-chance answers. See the combined report at
`artifacts/qwen05-long-retrieval-20261016-r1/FINAL-RETRIEVAL-REPORT.md` and
`final-retrieval-audit.json` for all four arms, both formats and limitations.
