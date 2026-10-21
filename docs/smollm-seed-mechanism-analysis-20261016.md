# Why seeds 8, 11 and 12 improved with virtual sinks

This analysis compares all six completed paired scratch-training experiments,
including the three non-improving pairs. It uses downloaded configurations,
all optimizer/evaluation records, final per-block losses, all-position gate
statistics, late-query diagnostics and sink-disabled evaluations. Evidence is
under `artifacts/smollm-seed-analysis-20261016/`; `analysis.json`, `curves.json`
and `source-breakdown.json` are reproducible summaries.

## Main finding

The evidence favors an **initialization-dependent change in optimization
trajectory**, with a modest direct inference-time contribution. Neither final
mean sink mass nor final sink gradient magnitude alone predicts improvement.
The strong statement that successful runs simply keep their sink gradients
alive is contradicted by seed13, which has comparable sink gradients but loses.

| Seed | Updates / warmup | Test CE gain, standard minus sink | PPL reduction | Gain retained with sinks disabled | Test blocks improved |
| --- | --- | --- | --- | --- | --- |
| 8 | 30 / 1 | +0.25809 | 22.75% | 98.95% | 99.46% |
| 9 | 30 / 1 | -0.00297 | -0.30% | not a gain | 42.19% |
| 10 | 30 / 1 | -0.00184 | -0.18% | not a gain | 43.99% |
| 11 | 70 / 10 | +0.46570 | 37.23% | 97.34% | 100.00% |
| 12 | 70 / 10 | +0.07581 | 7.30% | 86.58% | 93.41% |
| 13 | 70 / 10 | -0.08213 | -8.56% | not a gain | 6.05% |

Retained gain is `(CE_standard - CE_sink_disabled)/(CE_standard - CE_sink_enabled)`.
It is a descriptive fixed-checkpoint intervention, not a causal decomposition
of training. The comparisons validate identical initial shared weights and
training settings within each pair. My independent configuration comparison
found no unexpected within-pair differences.

## New checkpoint intervention: does preceding context matter?

Read-only Slurm array217197 completed all six diagnostic tasks successfully.
Each task loads the two final checkpoints for one seed with its original frozen
source, then evaluates32 evenly spaced held-out blocks (16,384 target tokens).
It compares full causal attention with self-only attention at every layer.
The self-only condition retains the current input token but removes every
preceding token from attention. Both conditions disable virtual sinks, avoiding
the additional confound of changing the virtual/physical normalization ratio
when the number of visible keys shrinks.

| Seed | CE increase after prefix deletion, standard | CE increase after prefix deletion, trained sink model |
| --- | --- | --- |
| 8 | +0.00017 | +0.40058 |
| 9 | -0.00064 | +0.00002 |
| 10 | +0.00038 | +0.00078 |
| 11 | +0.87126 | +1.39921 |
| 12 | +0.85763 | +1.18459 |
| 13 | +1.11672 | +2.05090 |

This strongly supports a distinct functional outcome for seed8: its sink model
has learned to benefit from preceding-token attention, while all five other
short-run checkpoints show almost no average CE benefit from preceding-token
attention on this probe. For these five, mean absolute block-level CE changes
are also below0.001, so the near-zero average is not hiding large positive and
negative block effects. The seed8 sink model worsens on all32 probe blocks.

An add-one-smoothed empirical training-unigram predictor has CE7.61058 on this
same sample, versus approximately7.666–7.675 for the short-run baselines and
non-improving sink models. Seed8's sink model reaches7.43073 (normal) and7.43571
with sinks disabled; deleting prefix attention raises it to7.83629. This is
consistent with escaping a low-order/frequency-fitting regime and learning
useful contextual features. It does not prove that the other models implement
an exact unigram predictor: self-only attention still permits current-token
conditioning, and equal losses do not prove equal predictions.

In the70-update cohort both arms of every seed depend on preceding context.
The probe therefore cannot explain every gain as a binary transition from
context-free to contextual prediction. Seed13 is a crucial counterexample:
its sink model is most sensitive to context deletion, yet is less accurate.
Context sensitivity alone is not evidence of better context use.

The intervention is off-distribution and changes attention mixing as well as
information availability. A large penalty can reflect useful context,
fragility to a changed attention pattern, or both. These32-block exploratory
results complement rather than replace the complete2048-block final tests.
Raw per-block results are `context-seedN.json`; `context-analysis.json` contains
the summary. No checkpoints or training outputs were modified.

## The three improving runs differ from each other

### Seed8: a new downward trajectory below a shared short-run plateau

All three 30-update standard models end at test CE7.685–7.691, and sink seeds9/10
remain there. Sink seed8 ends at7.432. This is not an unusually weak seed8
baseline. Validation CE drops below7.6 at update14 with sinks versus24 without,
and below7.4 at update24 only with sinks. The curves briefly reconverge around
updates9–10 before a sustained new separation. An early advantage alone does
not explain the final effect.

The early pre-clipping gradient peak is58.40 for standard and4.78 for sink.
By update15, seed8 still has89 heads with intermediate mean gates in validation,
versus12 and10 in sink seeds9/10. At the final full-test evaluation, the counts
are82,14,11 respectively. Here intermediate means physical gate between0.01
and0.99 averaged over test examples and positions; this is not a direct
measurement of the gate's derivative on each token.

Sink seeds9/10 develop nearly binary head means and very small sink gradients.
Their largest gradient spikes are69.31 and30.87, respectively. This is consistent
with premature gate saturation contributing to their failure to change the
learning trajectory, but does not prove the direction of causality.

### Seed11: the sink avoids a much slower baseline trajectory

The 70-update sink endpoints are similar: test CE6.743,6.768,6.809. Their
standard counterparts are7.209,6.844,6.727. Standard seed11 is unusually slow;
the sink model is comparable to the other sink models. Across these three seeds,
standard CE SD is0.2512 versus0.0334 for sink CE. This suggests reduced
initialization sensitivity in this small sample, not an established variance
reduction result.

Validation CE reaches7.4 at update21 with sinks versus52 without, and7.2 at29
versus59. At update70 the standard model is still improving, so finite-budget
optimization speed is a credible explanation. The data do not establish that
the standard model is trapped permanently or that the sink has a better
asymptotic optimum.

Final mean late-query sink mass is just0.195%, despite the largest improvement.
Across all positions and all2048 test blocks it is0.787%. During validation,
all-position mass was24.46% at update10,7.58% at15,3.26% at20 and0.77% at70.
That temporal pattern is consistent with a training scaffold that becomes less
necessary after the ordinary weights adapt.

### Seed12: a smaller, persistent optimization advantage with larger sink use

Final late-query mass is12.48%, and all-position mass15.03%, much larger than
seed11. Validation CE6.8 is first reached at update57 with sinks versus60
without. The train CE advantage0.07531 closely matches the test advantage0.07581.
This is a broad, modest fitting advantage:93.4% of test blocks improve. Its
benefit is not confined to a narrow domain.

Disabling sinks raises CE by0.01017, removing13.4% of the paired improvement.
Seed12 therefore has more material direct inference reliance relative to its
total gain than seeds8/11, while still retaining most of the gain without sinks.

## What the losing runs rule out

Seed13 has110 intermediate-mean heads and final sink gradient norm0.00160,
similar to winning seeds8/11/12 (0.00157/0.00191/0.00135). Its maximum gradient
is also reduced, from30.69 standard to7.34 sink, yet the sink model loses.
Therefore gradient smoothing and nonzero gate gradients are insufficient to
predict the final outcome.

Seed13 initially learns faster with sinks: validation CE7.4 is reached at
update27 versus34, and7.2 at34 versus39. The order reverses later: standard
reaches7.0 at44 versus46, and6.8 at53 versus59. The standard model eventually
overtakes the sink model and ends with the best test CE of the entire70-update
cohort (6.727), slightly better than the best sink endpoint (6.743).

Its sink-disabled CE is even worse,6.822 versus6.809 enabled. The direct sink
contribution is beneficial at its learned weights; the full paired training
trajectory is nevertheless worse. This distinguishes a locally useful module
from a globally beneficial training intervention.

## Attention mass is a poor scalar explanation

| Seed | Final late-query mass | All-position mass | Intermediate mean heads /270 | Final sink gradient norm |
| --- | --- | --- | --- | --- |
| 8 | 6.12% | 7.82% | 82 | 0.00157 |
| 9 | 5.02% | 5.63% | 14 | 0.00000754 |
| 10 | 3.79% | 3.97% | 11 | 0.0000354 |
| 11 | 0.195% | 0.787% | 40 | 0.00191 |
| 12 | 12.48% | 15.03% | 119 | 0.00135 |
| 13 | 2.61% | 4.31% | 110 | 0.00160 |

Late-query mass uses the final supplemental8-block diagnostic; all-position
mass uses the full2048-block test gate statistics. They should not be conflated.
The layer heatmap also shows no universal winning pattern. Seed12 strongly
gates early/middle layers, while seed11 ends with very little mass and seed8
contains both intermediate gates and some effectively suppressed heads.

Reducing physical prefix attention is also insufficient: losing short-run
seeds9/10 reduce mean late-query first-four-token mass from1.67%/1.71% to
0.212%/0.315%, yet obtain no material benefit. Winning seed12 instead increases
that mass from0.348% to0.485%. The benefits cannot be reduced to moving mass
away from the first few physical tokens.

## Why a zero-value sink can change training

For one head, let `Z=sum_j exp(q·k_j/sqrt(d))` and `s=q·w+b`.
The implemented zero-value sink gives exactly

`output_sink = g * output_standard`, where `g=Z/(Z+exp(s))`.

It provides a query-dependent multiplicative gate on the attention branch.
It can suppress unhelpful mixing while the other weights learn, changing the
residual stream, downstream gradients and future attention scores. Parameters
can retain the resulting useful representations after the gate is removed.
For fixed physical logits, `dg/ds=-g(1-g)`: saturation makes the sink-logit
gradient small. However mean gate statistics alone do not measure per-token
gradients, and Adam moments, weight decay, and coupled query/key updates also
influence parameter changes. Adam's moment normalization also means a smaller
raw gradient is not automatically a smaller parameter update. Pre-clipping
gradient peaks alone cannot establish the mechanism.

This interpretation is consistent with the literature on softmax sinks as
non-informative score absorption and on adaptive no-operation behavior, but
these papers do not establish the cause of our seed-specific results:

- Gu et al., [When Attention Sink Emerges in Language Models](https://arxiv.org/html/2410.10781v1).
- Fesser et al., [A Unifying View of Attention Sinks](https://arxiv.org/abs/2606.08105).

## Breadth and remaining uncertainty

All three winning seeds improve mean test CE in every source domain. FineWeb,
Cosmopedia, Python, math and StackOverflow gains are respectively:

- Seed8:0.234,0.274,0.419,0.276,0.274.
- Seed11:0.430,0.468,0.782,0.472,0.363.
- Seed12:0.076,0.067,0.104,0.055,0.099.

Seed13 is worse in every source. StackOverflow has only20 test blocks, so its
domain-specific estimates are particularly exploratory. Test blocks are not
independent initialization trials or independent datasets. Within-seed
bootstrap precision does not establish robustness across seeds.

The warmup experiment also changes data budget, training length and seeds.
Its70-update data order differs from the30-update order. It is not a causal
test of warmup alone. We have not independently varied base-weight seed and
sink-weight seed, so their contributions cannot be separated.

These are very early training experiments:31.5M and73.4M input tokens for a
134.5M-parameter model, approximately0.23 and0.55 tokens per parameter. The
results characterize early optimization under this recipe, not mature language
model performance. All endpoints are the fixed final update, not selected
best checkpoints.

The leading hypotheses are: (1) early adaptive gating changes the time at which
useful features form; (2) abrupt updates sometimes saturate gates before that
benefit occurs; (3) gating can also slow or redirect an already favorable
baseline trajectory, explaining seed13. These hypotheses can coexist.

A more specific candidate is that compulsory attention mixing in an immature
30-layer network suppresses useful distinctions in the residual stream. A null
gate can selectively attenuate that mixing until useful contextual features
form. This predicts higher cross-token representation similarity or less useful
attention-branch directions in the stalled models. We have not measured those
hidden-state properties, so "overmixing" remains a hypothesis, not a diagnosed
cause. The checkpoint probe supports the contextual-learning distinction for
seed8, but does not establish which training-stage change produced it.

## Most discriminating follow-up

Use the same base initialization, data sequence and learning-rate schedule in
a small factorial experiment: standard, learned query sink, a fixed scalar gate
matched to initial attenuation, and a temporary sink removed after a
predeclared early update. Independently vary sink initialization with the base
seed held fixed. Replay the known winning and losing seeds before expanding
the sample. Compare at matched token budgets and predeclared endpoints.

If a fixed gate reproduces the gain, dynamic query conditioning is unnecessary.
If a temporary sink retains it, an early optimization scaffold is supported.
If removing sinks during training destroys the gain while disabling them only
at the end does not, the mechanism depends on ongoing co-adaptation. Extend
matched baselines to test a speed advantage versus a lasting endpoint advantage.

These are proposed causal experiments; no new training was launched as part
of this analysis. A separate read-only GPU checkpoint probe tests sensitivity
to preceding-token attention and is recorded with its own job/evidence files.
