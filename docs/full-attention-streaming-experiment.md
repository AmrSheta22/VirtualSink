# Full-attention training and cache-eviction evaluation

Run root: `~/data/smollm-full8192-20260913`. The existing SWA release remains immutable.
Two fresh full-attention arms retain the same seed 8 shared initialization, data seed
42, 8192-token sequences, 128 sequences/update, 30 updates, FP32 AdamW recipe,
sink std 0.02/bias +4, and no auxiliary loss. The exact prepared SWA data is reused
read-only. Window 0 explicitly selects full causal attention. Inner attention
checkpointing bounds activation memory on K80; a real 8192-token backward and Adam
step for each variant gates submission execution.

The short recipe keeps its previous compressed schedule, including update 1 at LR
zero. This intentionally holds optimization fixed across the paired experiment;
it is not a claim that 30 updates constitute mature pretraining.

Both the existing SWA pair and new full-attention pair get a dependent GPU job:

1. Verify final completion, matching initialization/data/settings and final budgets.
2. Compare 8192-token cached logits against parallel logits, for FIFO 512 and full
   attention, with absolute RoPE indices and a documented 0.003 absolute tolerance.
3. Evaluate two disjoint 32768-token held-out packed FineWeb streams, teacher-forced.
   These are concatenated test documents, not claimed to be individual long books.
4. For each model compare full cache, FIFO 512, and prefix-retaining cache (4 initial
   tokens plus 508 recent, total 512). Also evaluate query_sink FIFO with sinks off.
5. Report per-token CE, 512-token position bins, perplexity, excess CE and perplexity
   ratio against the same model's full-cache reference. Report positions beyond 8192
   as extrapolation. The cache includes the current input token; first eviction is
   input position 512 (zero based), predicting token 513.
6. Save layer/head mean and maximum sink probabilities, sink and physical logit
   summaries, SVG heatmaps, loss curves, high/low query-token examples per head,
   and initial/trained sink parameters with per-head weight norms.

Full-cache evaluation of the SWA model is a counterfactual mask change, not its
native baseline. FIFO is its native mask. A virtual sink uses no physical KV slot.
Cached token evaluation does not recompute the last window from scratch. No RoPE
rebasing is used; prefix retention is a cache-policy control, not an exact
reproduction of StreamingLLM's positional treatment.

No fixed 30% head-use threshold, punctuation rule, or bias-sign rule defines success.
No NaN or perplexity >1000 is assumed: these immature models can already have
perplexity above 1000 before eviction. Compare losses on the same targets.

Completed policy/stream results are written atomically and reused on requeue.
The diagnostic job requeues between units after 18 hours. An interrupted unit is
recomputed. Training jobs retain the existing partial-gradient resume mechanism.
