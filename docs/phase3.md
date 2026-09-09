# Phase 3 model adaptation and uptraining

The Qwen2.5 integration is implemented and CPU numerically tested. No pretrained
uptraining run has completed, and no language-model quality result is claimed.
The Phase 1–2 report supports further investigation on a real language model;
its synthetic-task gains do not establish language-model gains.

## Compute status

On 2026-09-09, local `nvidia-smi` reported a GTX 1650 with 4 GB VRAM.
A read-only `sinfo` query reported only `gpu:tesla_k80:2` nodes in the Bibalex
GPU partition, with no current jobs for this account. K80 does not support the
BF16 path used here. No GPU job was submitted. Obtain a working modern GPU
allocation, install a separate modern environment, and confirm actual GPU test
output before training. Follow [the HPC guide](../hpc/README.md); the existing
dirty restructuring has not been committed or pushed by this change.

Start with **Qwen/Qwen2.5-0.5B**, attention-only training. Qwen2.5-1.5B is also
supported once measured memory permits it. Llama is not implemented. The adapter
is pinned to Transformers 4.51.3 and preserves pretrained projections, their
biases, grouped-query attention, and rotary embeddings. Sink scores use queries
after rotary embedding. This positional choice needs evaluation; it is not a
claim of streaming or cache-eviction robustness.

The implementation uses FP32 master parameters and Adam states with BF16
autocast. Full training therefore requires approximately **16 bytes per
parameter** for parameters, gradients and Adam moments, before activations and
temporary allocations: about 24 GB for 1.5 billion parameters. Attention-only
training reduces optimizer/gradient storage, but frozen weights still occupy
memory. Weight-only BF16 estimates are not full-training memory estimates.

The backend is query-chunked, checkpointed dense attention, **not FlashAttention
or fused SDPA**. It retains quadratic arithmetic and repeats GQA keys/values.
Decoder checkpointing and vocabulary-chunked cross-entropy reduce activation
storage. No production memory/throughput result is available. Benchmark 20–100
steps in the actual allocation before choosing microbatch size or extrapolating
runtime. Default microbatch 1 with accumulation 32 is conservative; try 4 with
accumulation 8 only after measurement. Both give effective batch 32.

## Objective and controls

The selected objective is causal LM cross-entropy plus
`0.01 * mean_layers_heads(-log(mean_batch_tokens(G) + 1e-6))`.
It discourages heads whose mean physical attention mass approaches zero. It
does not equalize head usage and is not a conventional MoE load-balancing loss.
The user explicitly selected this change from the proposed entropy-like term.
The head averages are computed per **microbatch**, then penalties are averaged
over accumulation steps; this is not the nonlinear penalty of a global batch
mean. Keep microbatch size matched between compared runs.

`--auxiliary supplied` retains the original formula for an explicit ablation;
minimizing that formula permits both near-zero and near-one physical mass.
`--auxiliary none` removes the penalty. Validation records every layer/head's
mean physical mass and the number below 0.01; inspect these alongside CE.

Option A (`--mode attention`) trains Q/K/V/O projections, including Q/K/V biases,
and sink weights/biases. MLP, embeddings, output vocabulary projection and norms
are frozen. Option B (`--mode full`) trains the whole backbone. Base learning
rate is 2e-5, sink rate 1e-4, with cosine decay, a warmup capped at 100 steps,
gradient clipping at 1, and AdamW weight decay 0.01 for base parameters and zero
for sinks. Nonfinite losses or gradients stop execution.

Initial validation measures the unmodified-attention control through the same
backend and the inserted-sink model. Run `--variant standard` in a separate fresh
directory for a **matched continued-pretraining control**, using the same model
revision, data, seed, budget and freeze mode. Sink parameters are disabled/frozen
and the auxiliary term is zero in that arm. Compare held-out CE/perplexity, not
the composite training loss. A single query-sink run cannot establish a benefit.

## Data preparation

Prepare disjoint train/validation/test JSONL files containing `{"text": "..."}`
records from a suitably licensed, cleaned and deduplicated source. FineWeb-Edu
or SlimPajama are suitable candidate sources; this command does not download or
claim to clean them. Deduplication and document-level split separation must be
established upstream. File-hash checks reject identical splits, not overlapping
documents. Pin the source revision and record its license and split policy.
WikiText-103 is too small for a unique billion-token subset; the runner refuses
silent corpus repetition.

Install into the separate environment:

```bash
python -m pip install -e '.[uptraining,dev]'
python -m virtualsink.uptraining.data \
  --train /data/train.jsonl --validation /data/validation.jsonl --test /data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B --revision MODEL_COMMIT_SHA \
  --source 'DATASET REVISION LICENSE AND SPLIT POLICY' --output-dir /data/phase3-tokens
```

Replace `MODEL_COMMIT_SHA` with the actual immutable 40-character Hugging Face
commit. Packing appends EOS between documents, allows cross-document attention,
uses no padding, and drops incomplete final blocks separately per split.
There is no shuffle inside training: shuffle documents reproducibly upstream.
Manifests record model/tokenizer revision, source/file hashes and token counts.
The runner verifies hashes and token ranges before training.

At sequence length 2,048 and effective batch 32, 15,000 optimizer steps consume
983,040,000 input tokens; 30,000 consume 1,966,080,000. Three billion tokens need
about 45,777 steps. Causal CE scores 2,047 targets per sequence; no loss crosses
the boundary between sequence blocks. A billion-token uint32 stream is about
4 GB on disk, excluding source text and checkpoints.

## Running and resuming

Configure the partition/account/GPU request in `hpc/train_phase3.sh` for an
available BF16-capable allocation. Its current inherited Bibalex partition is
K80-only and will fail the explicit prerequisite. Set `PHASE3_ENV` and
`PHASE3_DATA`, commit/push the reviewed code, and submit through the existing
workflow. Do not install packages or run training on the login node.

Inside the allocation, the equivalent command is:

```bash
python -u -m virtualsink.uptraining.train \
  --data-dir /data/phase3-tokens --output-dir runs/phase3-query \
  --mode attention --variant query_sink --auxiliary anti_collapse \
  --steps 15000 --seq-len 2048 --batch-size 1 --accumulation 32
```

Use the same arguments and `--resume runs/phase3-query/last.pt` to resume. The
model, Adam state, scheduler, step, CPU/CUDA RNG and best validation CE are saved
atomically every evaluation interval (default 250 steps). Data order derives
from the optimizer step, so interrupted accumulation is replayed from the last
checkpoint. Configuration, code hashes and corpus must match exactly. Resume
only trusted checkpoints; optimizer snapshots use PyTorch pickle loading.
Resuming an older checkpoint appends logs and can produce repeated step numbers.

`best.pt` minimizes validation CE; `last.pt` supports restart. `summary.json`
contains test metrics from the best checkpoint. Validation/test use up to
`--eval-batches` fixed batches (32 by default), and record the evaluated sequence
count. Increase this fixed evaluation budget before a confirmatory study.
Checkpoint files include full FP32 model and optimizer state, so reserve disk
for best, last, and an additional atomic temporary checkpoint. These are custom
checkpoints: reconstruct Qwen, call `adapt_qwen`, then load `model` state; standard
AutoModel loading alone does not install the sink forward method.

The adapter explicitly rejects padding, sliding windows and KV-cache use.
Its `output_attentions` entries contain physical gates `(B,H,T)`, not dense
attention matrices. Generation, cache eviction and fused kernels remain future
work. CPU tests use randomly initialized tiny Qwen configurations and synthetic
tokens; they do not count as Phase 3 uptraining results.

References: [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[PyTorch SDPA API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
