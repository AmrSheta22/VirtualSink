# Uptrained streaming evaluation

Evaluate the completed regularized SWA step-1000 standard/query-sink pairs for
Qwen2.5-0.5B and SmolLM-135M. This is separate from other scratch-pretraining
streaming experiments and from the cancelled no-auxiliary Qwen run.

Each model scores the first 8192 next-token targets from two individual WikiText
official test articles: American Beauty (1999 film) and 2003 Pacific typhoon
season. Articles are reconstructed at top-level headings; no cross-article
concatenation. Text and source hashes and available token lengths are recorded.
Tokenizations differ between models; comparisons are within-model.

Four conditions: standard FIFO512, standard prefix4+recent508, trained sinks
FIFO512, same sink checkpoint FIFO512 with virtual sinks disabled. Stored KV
contains at most 512 physical tokens per layer, including the current token;
virtual sinks consume no physical cache slot. The first eviction is zero-based
input position512, predicting target513. No hidden-state/window recomputation
or cache resets occur inside an article. Cache tensors are capped FIFO storage,
not a specialized performance-optimized ring buffer.

Absolute RoPE positions are retained in every condition. Prefix retention is a
cache-policy control, not an exact implementation of StreamingLLM's positional
rebasing. The learned query sink receives the same rotated query used in native
training; do not silently rebase it. Positions beyond1024 exceed uptraining
sequence length; this is part of the test, and can independently affect results.
No full-context reference is included in this first bounded-cache experiment.

Before evaluation, cached logits must match parallel full and window512 logits
on641 tokens, crossing eviction. Absolute tolerance0.003, relative0.0001 are
checked per logit; maximum absolute errors are saved. Tiny-model tests cover
Llama and Qwen projection bias, GQA, enabled/disabled sinks, full and rolling
attention, and exact prefix preservation. A failed check stops evaluation.

Outputs: per-token CE,128-token loss bins, per-article curves, before/after
layer-head heatmaps, per-token sink probabilities/logits/physical LSE in NPZ,
high/low query-token examples per head, original and learned sink parameters,
and comparisons against the standard prefix-retaining control. Token associations
are descriptive, not causal claims of idling. Completed units save immediately;
progress JSON refreshes every512 tokens. No automatic resubmission is enabled.

Success is reduced post-eviction degradation relative to standard FIFO, with
performance approaching the retained-prefix control at equal cache budget.
Neither perfectly flat loss, collapse at exactly w+1, nor NaN is assumed.
