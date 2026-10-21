# StreamingLLM replication screen and initialization pilot

14 October scope update: the user cancelled both recomputation controls217043_2/5 and requested streaming evaluation of the trained Pythia410M baseline/virtual-sink pilots. See [the new protocol](pythia410-trained-streaming-20261014.md). Original pretrained FIFO/StreamingLLM tasks continue; the historical six-unit completeness requirement below is superseded. Current replay-and-stream array217086 and combined report217087 replace report217045, which was cancelled while pending. First replay array217080 failed before training and is preserved.

Current repair attempt: `/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3`, reusing immutable prepared data/upstream/model snapshots from the original root. Preparation217041, GPU validation217042, evaluation array217043, initialization pilot217044, report217045. CPU preparation/report allocations have been retargeted from the busy comp059 node to comp067. Source and dependencies are isolated in the r3 release.

Authorized 13 October 2026 by the user's annotated request to start the three research steps. This is a staged screen, not a completed four-million-token replication.

Remote root: `/cluster/users/alex116u1/data/streamingllm-replication-20261013`. Local frozen source and scripts: `artifacts/streamingllm-replication-20261013/`. Remote jobs.tsv is authoritative.

## Pretrained diagnostic

Targets are EleutherAI/pythia-2.8b (a model size evaluated in StreamingLLM) and the original unmodified Qwen/Qwen2.5-1.5B checkpoint. Pythia's model revision and deduplication choice are recorded explicitly; the model used here is the non-deduplicated release. This is not a claim that the paper's exact unpublished checkpoint identity has already been matched. Qwen is a generalization diagnostic, not a paper-model replication.

Each model scores 400,000 PG-19 predictions under FIFO/absolute RoPE and StreamingLLM/prefix-four/cache-relative RoPE. A recomputation control scores the identical first 20,000 predictions, with fresh hidden states over the recent window. All comparisons against recomputation must use this shared first-20k range. Cache budget is 1,024 saved physical entries; matching the released evaluator, the current prediction transiently attends to up to 1,025 entries, then trims back to 1,024. This differs from the earlier Qwen512 protocol, which counted the current token inside the attention budget.

The PG-19 test filenames are read from an immutable DeepMind dataset revision and lexicographically sorted, matching its loader. Each book is tokenized separately, with persistent cache between books, no extra injected EOS, and no scored prediction from the last token of one book to the first token of another. This matches the official released evaluator's book-boundary behavior. Book IDs/text hashes, model/tokenizer revisions and prepared-data hashes are recorded. Token perplexity is compared within a model, not across different tokenizers. This corpus is a replication benchmark, not asserted to be absent from model pretraining.

Pythia uses FP16 pretrained weights on K80 and the pinned backend's attention-score normalization. The independent GPTNeoX streaming decoder supports partial RoPE and parallel residuals. The validation gate compares tiny-model parallel logits, official raw-key shifting across repeated eviction, then full Pythia2.8B cached and official post-eviction logits. Qwen uses the existing FP32 native implementation with sinks disabled, verifies parallel/window parity and an independent delta-rotation oracle, including the official post-prediction trim convention. No uptraining weights are used in these diagnostic arms.

## Initialization pilot

EleutherAI/pythia-410m has three fresh seed1729 arms: standard, legacy query-conditioned zero-value sinks (score weight std1e-4/bias-3.5), and calibrated near-identity sinks (score weight zero, per-head bias derived from minimum physical logsumexp on four training blocks, targeting initial sink mass at most1e-5 on those samples). Calibrated logits must differ from disabled/pretrained logits by relative L2 less than0.001. This is approximate function preservation on calibration samples, not an exact guarantee on every future query. An exactly zero multiplicative gate could starve score gradients, so the sink starts with small positive mass instead.

All arms use PG-19 training books and a separate validation split, the same ordered token streams, full causal512-token training blocks, attention projections only, FP32, no auxiliary loss, AdamW with base/sink peak2e-5/1e-4, 25-step warmup and cosine decay, microbatch1/accumulation8, 256 steps and1,048,576 input tokens. No corpus repetition is allowed. Validation uses32 packed512-token blocks every32 steps. This is a one-seed pilot, not a complete tuning study. It stores only changed parameters against an immutable pretrained model. It does not yet establish streaming robustness of the trained pilot checkpoints.

## Jobs

- 216946: CPU preparation, isolated transformers4.33.3/tokenizers0.13.3 target directory; existing environments unchanged. Pin upstream StreamingLLM commit, download pinned models and PG-19 data.
- 216947: GPU validation, after successful preparation. Exercise official decoder parity and the pilot's attention/gradient path before allowing experiments.
- 216948: six evaluation tasks, concurrency3, after validation. Indices0/1/2: Pythia FIFO/StreamingLLM/recomputation; indices3/4/5: Qwen equivalents.
- 216949: pilot tasks0/1/2: standard/legacy/calibrated, concurrency2, after validation.
- 216950: CPU completeness and comparison report after both arrays terminate; missing outputs yield INCOMPLETE.json and no complete campaign conclusion.

GPU allocations exclude comp013/comp016. Evaluation allocations have48-hour limits; progress is written every512 predictions. These screen jobs do not automatically resume or requeue: timeout or failure must be reported, not silently treated as a result. Pilot allocations have24-hour limits. All batch scripts source an isolated hpc/hpc_runtime.sh. Long computations and dependency installation occur inside Slurm allocations.

Sources: [StreamingLLM paper](https://arxiv.org/html/2309.17453v4), [official evaluator](https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py), [official positional shift](https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/pos_shift/modify_gpt_neox.py), [PG-19 loader](https://huggingface.co/datasets/deepmind/pg19/blob/main/pg19.py).

The first submission (216941–216945) terminated before application logs on CPU node comp090 with signal53. No evaluation or pilot training executed. Job records are preserved in jobs-first-attempt.tsv. The replacement CPU preparation/report scripts are pinned to comp023, which successfully ran earlier campaign reports; the authoritative replacement IDs are listed above.

Preparation216946 started successfully on comp059 after the pending resource request was adjusted to2 CPUs and retargeted from the busy pinned node. Report216950 is also retargeted to comp059 with2 CPUs/8GB. These are recorded Slurm resource overrides; running scientific source is unchanged. Startup and environment-ready logs were inspected more than20 seconds after launch.

## Dependency repair — 13 October 2026

Preparation216946 completed in10m26s and all assets remain available. Validation216947 failed during Transformers import because the partial `--no-deps` installation omitted regex. No evaluations or pilot steps ran; the first report correctly recorded INCOMPLETE. The r2 repair installs the complete pinned dependency set into a fresh target directory, resolves transitive dependencies, stores dependencies-lock.txt, and imports GPTNeoX/Transformers in a fresh process before GPU validation. It verifies every prepared-data checksum and preserves the original failed release.

The r2 complete dependency check passed in1m28s and tiny-model checks passed. Validation217037 then rejected the custom Pythia2.8B cached decoder against the FP16 HF reference (maximum mismatching logit0.04297, tolerance0.03). No evaluation or pilot ran. The r3 decoder reuses the pretrained backend's cached rotary values and its stock fused attention/scaling path instead of regenerating/scaling them differently. Validation tolerances are unchanged. A separate frozen r3 release preserves the r2 failure and reuses the same pinned assets.

The r3 repair passed CPU verification in46 seconds and GPU validation217042 in15m22s, finishing13 October21:25:38 Cairo. Pythia pre-eviction maximum logit error was0; official post-eviction parity passed over1153 tokens. Qwen maximum errors were0.0007615 for parallel parity and0.0004530 for the independent rebasing oracle; the post-prediction trimming comparison also passed. Evaluation tasks217043_0/1/2 and pilot tasks217044_0/1 started on compute nodes, with startup/environment logs inspected more than20 seconds after launch. Remaining array units wait for their concurrency limits. The three-hour monitor follows r3.
