# SmolLM 8192 / SWA 512 restart

This supersedes the cancelled full-attention 2048-token experiment. Both arms
start from identical random shared weights (seed 8), with the same prepared data
(seed 42), 30 updates, no auxiliary loss, and full-model FP32 optimization.

- Sequence length: 8192; causal window: 512 including the current token, every layer.
- Global batch: 128 sequences, preserving 1,048,576 target tokens per update.
- Sink weight initialization: Normal(0, 0.02); sink bias: +4.0. Baseline disables sinks.
- RoPE theta remains 10000; maximum position configuration is 8192.
- Training attention slices keys/values to the union of each query chunk's windows,
  then applies exact per-query masking. The virtual sink remains available everywhere.
- Validation and attention diagnostics every update; microbatch progress every eight
  sequences; resumable checkpoints at least every ten minutes.
- Diagnostics respect the mask and additionally report oldest-visible-token mass.
  Absolute first-token mass becomes zero once that token leaves the window.
- Comparison rejects missing hashes, different data/recipe settings, unequal completed
  budgets, and invalid paired loss vectors. Existing prepared data is validated before reuse.

This is an intentional modification of the SmolLM recipe, not an exact reproduction:
longer sequences, SWA, FP32 native backend, shortened schedule, public data subsets,
and the documented StackOverflow/Python-Edu substitutions. The scaled schedule keeps
its initial zero-learning-rate update. Shared initialization is independent of sink draws.

The isolated HPC root is `~/data/smollm-swa8192-20260911`. Previous outputs remain in
`~/data/smollm-scratch-20260910`. New data is generated with the pinned tokenizer and
source revisions. It packs EOS-separated documents; attention can cross EOS boundaries.
