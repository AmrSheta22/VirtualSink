# Phase 3 K80 pilot protocol

Two user-authorized runs use FP32 on one K80 GPU each:

| Model | Trainable parameters | Input tokens | Steps | Sequence | Microbatch / accumulation |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM-135M | Full backbone and query sinks | 8,192,000 | 1,000 | 256 | 1 / 32 |
| Qwen/Qwen2.5-0.5B | Q/K/V/O projections and query sinks | 8,192,000 | 1,000 | 256 | 1 / 32 |

Both use causal cross-entropy plus the user-selected anti-collapse penalty
(gamma 0.01), base LR 2e-5 and sink LR 1e-4, cosine decay with 100-step warmup,
gradient clipping at 1, checkpointed attention/decoder/CE, and fixed seed 1729.
Actual trainable and total parameter counts are printed at model load.

The small model is pretrained SmolLM-135M, not a random model trained from scratch.
Both start from immutable Hugging Face model commits resolved during preparation.
The PyTorch 1.12 native decoder retains safetensors parameter names, SwiGLU MLP,
RMSNorm, GQA and RoPE. Tests compare original-attention CE and gradients against
Transformers for tiny Llama and Qwen configs. Original pretrained tensors must
load without unexpected/missing keys other than newly inserted sinks.

## Data and comparisons

Preparation downloads a pinned revision of Salesforce/wikitext, configuration
wikitext-103-raw-v1. Official train/validation/test splits remain separate. It
reads the same ordered, nonempty source rows for both tokenizers and appends EOS
after each row. This is row packing, not reconstructed Wikipedia articles.
It stops after both tokenizers have enough training tokens; each run consumes
its first 8,192,000 tokens. The exact text boundary can differ because tokenizers
differ. Validation/test preparation retains at least 131,072 tokens each; the
pilot evaluation scores the first 32 sequences (8,160 prediction targets) per
split. Source revisions, parquet hashes, row counts and token-stream hashes are
saved in each manifest. Data is not downloaded from, or uploaded to, private sources.

Initial unmodified-attention validation and inserted-sink validation are logged.
Every 25 steps, CE/perplexity and mean physical gates for every layer/head are
recorded. Best-checkpoint test evaluation occurs after training. Perplexities
across different tokenizers are not directly comparable. These two runs establish
feasibility; neither alone isolates sink benefits from ordinary continued
pretraining. Matched trained standard-attention controls remain a later experiment.

## Checkpoint and output policy

Each job has its own run directory. JSONL logs flush each row and contain UTC
timestamps, optimizer/microbatch progress, CE, auxiliary loss, gradient norm,
learning rates, throughput, peak allocated/reserved GPU memory, evaluation
metrics, checkpoint durations and sizes. Slurm stdout/stderr are separate files.

Initial `best.pt` and `last.pt` are saved before optimization. A fresh `last.pt`
is written every 10 completed optimizer steps or after 600 seconds, checked at
optimizer-step boundaries. Its predecessor becomes `previous.pt`. Validation
improvements update `best.pt`. Milestones at steps 250/500/750/1000 are retained.
Checkpoints include full FP32 model, Adam and scheduler states, CPU/CUDA RNG,
data/model manifest, source hashes, step and best validation CE. The temporary
file completes before replacing a checkpoint; previous/last protect restart.
Logs and checkpoints stay on shared HPC storage through job exit.

The job reserves 24 hours and asks the trainer to checkpoint/pause after 23.5
hours, or at the next optimizer boundary following a termination signal. A
Slurm USR1 signal is requested five minutes before the allocation ends. No
guarantee can be made for a hard kill during a long optimizer step. Resume uses
`last.pt` with unchanged code/config and deterministic token offsets.

## Timing

24 hours is an allocation ceiling, **not a runtime estimate**. The first two
optimizer steps are excluded from timing estimates. Starting after five completed
steps, `timing.json` reports the observed mean, min/max-based remaining range,
and measured validation/checkpoint overhead. Estimate quality improves after
20–100 steps. JSONL timing events are emitted every ten steps. These ranges are
descriptive, not confidence intervals; hardware contention and storage variability
can change them. Queue, downloads and startup are explicitly excluded. Final
elapsed times must come from completed logs and Slurm accounting.

## Deployment

The working repository and remote checkout already contain unrelated local
changes. Deployment therefore uses a dedicated Git branch and a detached remote
worktree at the committed release, without resetting either checkout. Preparation
creates a separate environment inheriting the verified `venv-k80` package path;
it does not replace the base Torch install. CPU preparation must succeed before
either GPU training job runs (`afterok` dependency). Each GPU job repeats a real
CUDA forward/backward probe before loading its model.

The 2026-09-09 GPU access probe **214314** completed successfully on **comp008**
with PyTorch 1.12.1+cu113 and a Tesla K80 reporting 11,441 MiB. This supersedes
the older blanket GPU-access failure status; comp013 and comp016 remain excluded.

Model reference: https://huggingface.co/HuggingFaceTB/SmolLM-135M
Dataset reference: https://huggingface.co/datasets/Salesforce/wikitext
