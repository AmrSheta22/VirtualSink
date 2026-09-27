# Matched SmolLM-135M from-scratch experiment

This experiment tests whether virtual sinks available from random initialization
change the emergence of physical sinks. It does not load or re-use pretrained
model tensors. Both arms initialize every shared tensor with model seed 8 and
verify the same SHA256 of the initialized shared weights. Data seed is 42.
The standard arm disables and freezes sink parameters; the query arm uses the
existing query-conditioned zero-value sink with bias -3.5 and weight std 1e-4.
Both optimize causal cross-entropy alone, with no auxiliary penalty.

## Source of the recipe

The authoritative recipe is the [official SmolLM1-135M YAML](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm1/config_smollm1_135M.yaml).
Initialization and optimizer conventions were checked against release-era
[Nanotron](https://github.com/huggingface/nanotron/tree/59bfb6b38c4487b04e425d8540b6e44b2a7fbcf9).
This is a public-recipe reconstruction, not an exact reproduction of the original
distributed run, unpublished token order, or original random-number stream.

| Setting | Paired experiment |
|---|---|
| Architecture | 30 layers, width 576, MLP 1536, 9 query / 3 KV heads |
| Vocabulary / tokenizer | 49,152; cosmo2-tokenizer, immutable revision |
| Context | 2,048 inputs and 2,048 shifted targets; full causal, no SWA |
| Embeddings | Tied input embedding and LM head |
| Initialization | Normal std 0.0416; output/down projections divided by sqrt(60); RMSNorm weights 1 |
| Global batch | 512 sequences = 1,048,576 predicted tokens per optimizer step |
| Optimizer | AdamW, LR 0.003, betas 0.9/0.95, eps 1e-8, decay 0.01 |
| Decay exclusions | None, matching release-era Llama/Nanotron |
| Clip | Global gradient norm 1 |
| Schedule | Linear warmup, plateau, 1-sqrt cooldown to zero |
| Mixture | 70% FineWeb-Edu, 15% Cosmopedia, 8% Python-Edu, 6% OpenWebMath, 1% StackOverflow |

The YAML has 600,000 original steps, 5,000 warmup steps, decay starting at
500,000 and lasting 100,000. The shortened run scales these boundaries and
rounds to integer steps (minimum warmup one step). This is a final-one-sixth
cooldown, unlike the blog's rounded 20% statement. As in Nanotron LambdaLR,
the first update uses LR zero; warmup and decay are indexed by completed steps.

## Explicit departures

- Fewer optimizer steps, with the warmup/plateau/cooldown boundaries scaled to
  the chosen budget. This is a pilot, not a fully pretrained language model.
- FP32 on K80 instead of BF16. Microbatch 1 and accumulation 512 replace
  distributed microbatch 8, accumulation 2, data parallelism 32. Effective
  batch and loss weighting are preserved; floating-point reduction order differs.
- The native, numerical-parity-tested decoder replaces Nanotron fused kernels.
  Query-chunk and decoder recomputation bound memory; arithmetic remains dense.
- Public pinned corpus subsets replace the unavailable original tokenized
  corpus/order. Shards and the mixed block order are deterministically shuffled.
  The exact rounded sequence quotas match the published weights.
- The original YAML does not identify its StackOverflow snapshot. The user
  authorized `bigcode/stackoverflow-clean` as the explicit 1% substitute.
- SHA256 of stripped document text assigns 98% training, 1% validation, 1% test,
  consistently across all components; an identical document cannot enter two
  splits. This does not guarantee semantic/near-duplicate separation.
- Supplemental validation/test CE and attention measurements replace the
  original large downstream benchmark evaluation suite in this pilot.

## Checkpoints and measurements

Use a fresh isolated root, never an earlier uptraining directory. Data preparation
runs on the CPU partition. Both GPU jobs depend on successful preparation and
are independent of each other. Each starts with a real CUDA forward/backward
probe and uses the same immutable release.

Every ten optimizer steps, measure fixed held-out CE and per-layer/head physical
gates. Supplemental attention measurements use late queries (positions 1024-2047)
and report first-token, first-four-token and virtual probability, virtual-logit
minus physical-log-partition, and first-four majority frequency. All masses use
the joint physical-plus-virtual denominator. Step logs include sink-gradient norm.

The primary comparison is the final matched optimizer step on the same 128 test
blocks. Best validation checkpoints are retained separately. The virtual arm
also evaluates its final weights with sinks disabled, without further training.
Do not equate a disabled-sink intervention with a separately trained baseline.

`last.pt` records model, Adam moments, scheduler, completed steps, partial
accumulation count and gradients, CPU/CUDA RNG, and configuration/source hashes.
It is atomically replaced at least every ten minutes at microbatch boundaries,
and after each completed optimizer step. The run checkpoints and exits 75 before
the allocation limit; the launcher requeues that SAME job ID. A pause never
looks like successful completion to downstream dependencies. Nonfinite losses
and gradients fail the job rather than automatically requeueing.

The trainer saves `best.pt`, `last.pt`, and `final.pt`, plus JSONL metrics,
configuration, status, and final summary. Initial weight hashes in the two
configs must match before interpreting the pair. Runtime estimates from early
microbatches include training only, and explicitly exclude queue/setup/evaluation
and checkpoint overhead.
