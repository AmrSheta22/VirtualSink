# Qwen2.5-1.5B matched uptraining seeds on Bibalex

Submitted 10 October 2026 (Africa/Cairo) in response to the larger-model replication request.

## Scope change: queued seeds cancelled

On 10 October 2026 at approximately 14:16 Cairo, the user cancelled the queued runs. Array tasks 216613_2, _3, _4, and _5 (seeds 1730 and 1731, both variants) were cancelled with a PENDING-state filter; Slurm confirmed CANCELLED with zero runtime. The seed-1729 standard and query-sink tasks 216613_0 and _1 remain running. The three-hour monitor now follows only that pair and excludes the intentional cancellations from failures and completion estimates. The original design below is retained as deployment provenance; the remaining experiment has one seed pair, not three.

## Fixed design

Three seeds, 1729/1730/1731, each receive two runs starting from the same pinned pretrained Qwen2.5-1.5B backbone: standard attention and query-conditioned virtual sinks with anti-collapse auxiliary loss (gamma 0.01). The standard arm disables/freezes sinks and effectively uses no auxiliary loss. Seed variation changes sink initialization; the pretrained backbone, ordered data, and dropout-free computation otherwise remain fixed. Thus the standard arms can be identical or nearly identical; these are not independent random pretrained backbones.

Each run uses attention-projection updates, FP32 on one K80, 1,024-token packed sequences, SWA 512 including the current token, microbatch 1, accumulation 8, and 1,000 optimizer iterations (8,192,000 input tokens). Base/sink peak learning rates are 2e-5/1e-4 with 100-step warmup and cosine decay. Validation uses the first 32 blocks every 25 iterations; lowest validation CE selects the checkpoint. The trainer performs its existing 32-block final test.

The model revision resolves once during preparation and is recorded in model-identity.json. Model identity is Qwen/Qwen2.5-1.5B at 8faed761d45a263340a0528343f099c05c9a4323. Token streams reuse the pinned original Qwen0.5B WikiText subset only after exact parsed tokenizer.json equality, EOS parity, and stream SHA256 verification. No corpus repetition is allowed.

This is a larger-model replication using the current gradient-preserving native implementation, not a byte-for-byte replay of the September releases. Source is frozen in the isolated release with source-sha256.json. The current native gate-gradient fix is included. Query and vocabulary chunks are 32 for memory; the training objective and token budget remain matched across all six arms.

## Jobs and execution gates

- Preparation: 216611, CPU; downloads pinned weights and verifies token compatibility.
- GPU validation: 216612, depends on successful preparation; runs the native/SWA/streaming tests and real-model auxiliary-only gradient checks, followed by two accumulated AdamW steps per variant.
- Training array: 216613, indices 0–5 with concurrency 2; depends on successful GPU validation. Indices 0/1 are seed 1729 standard/sinks; 2/3 are seed 1730; 4/5 are seed 1731.

All jobs source an isolated hpc/hpc_runtime.sh, activate the existing verified PyTorch 1.12.1+cu113 runtime, and log heartbeats. GPU jobs exclude comp013 and comp016. Training allocations request 48 hours and 32 GB host RAM. The trainer saves/pause-checks at optimizer boundaries, with a 47.5-hour budget and Slurm advance signal forwarding; a successfully checkpointed unfinished run requeues and resumes with exact configuration equality. Failures do not automatically retry.

Checkpoint cadence is 25 steps or 1,800 seconds, plus best checkpoints and the step-1,000 milestone. last.pt and previous.pt preserve restart state. Existing unrelated jobs and environments are retained.

Remote root: `/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010`.

Local deployment provenance: `artifacts/qwen15-matched-seeds-20261010/` (plan.json, source-sha256.json, jobs.tsv, and release.tar.gz).

## Interpretation

These runs replicate the original regularized query-sink versus standard-attention comparison at a larger model size. They do not train the additional no-auxiliary and bias-only arms. Persistent streaming evaluation and the rebased-prefix baseline must be run after training; the trainer's short packed-block test does not establish streaming robustness.

Do not treat six runs as six independent data samples. Summarize effects per matched seed pair and per held-out document, report individual seed outcomes, and explicitly acknowledge that deterministic standard arms may coincide.
