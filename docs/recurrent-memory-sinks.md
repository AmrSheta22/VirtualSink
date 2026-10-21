# Eviction-written LSTM virtual memory — implementation and pilot protocol

This backend implements one zero-value null sink and three dynamic memory slots
per query head, at every native Qwen/Llama decoder layer. It is separate from the
existing static-value sink experiments. No model-quality improvement is claimed
by implementing the mechanism or passing its correctness gate.

## Architecture

`src/virtualsink/uptraining/recurrent_sink.py` wraps a `NativeLM` without replacing
its attention modules. The original static `sink_w/b/v` are unused and frozen in
the wrapper. The wrapper owns a new query-conditioned null score and recurrent
memory parameters; the pretrained tensors remain intact on construction.

Each stream carries raw physical K/V plus three FP32 cell/exposed state pairs
`[B, H, 3, D]` per layer. Nothing is written before the physical cache fills.
At position `t >= capacity`, the token at `t-capacity` is written once, before
attention reads the retained physical window and updated memories. The current
token counts toward physical capacity, matching the existing FIFO convention.

The write input is layer-normalized concatenated **unrotated key and value** of
the evicted token. GQA K/V is repeated to its associated query heads. Gates use:

```text
[f, i, o, g] = write_input @ W + exposed @ U + gate_bias[head,slot]
cell_next    = sigmoid(f) * cell + sigmoid(i) * tanh(g)
exposed_next = sigmoid(o) * tanh(cell_next)
```

The two transformation matrices and memory read projections are shared across
heads and slots within a layer. Gate biases and states are separate for each
head/slot. This gives Qwen2.5-1.5B (28 layers, 12 heads, head dimension 128) about
7 million new parameters, versus about 232 million for dense independent
head/slot matrices. Separate retention biases break slot symmetry after writes;
the gates can learn head/slot-specific policies through their biases and states.

Memory keys and values are projected from `exposed_next`. They use the raw
current query and a scaled dot product; they receive no absolute RoPE position.
Physical attention uses ordinary absolute RoPE. The null sink retains the
existing affine function of the rotated query, with no extra dot-product scale.
Physical, null and memory logits share one softmax. Null contributes zero value.

The state is a content compression mechanism, not lossless archival storage.
Three vectors cannot retain arbitrary old facts indefinitely. A high forget
gate does not guarantee useful memory, and repeated writes can overwrite it.

## State and differentiability

```python
from virtualsink.uptraining.native import load_native
from virtualsink.uptraining.recurrent_sink import RecurrentSinkLM

model = RecurrentSinkLM(load_native(pretrained_directory), capacity=512).cuda()
groups = model.parameter_groups(mode="memory")  # or attention/full
result = model(segment_tokens, state=None)
next_result = model(next_segment_tokens, state=result.state)
# Compute loss/backward across the unrolled segments, update parameters, then:
state = next_result.state.detach()
# At an authoritative document boundary:
state = None
```

`forward()` returns hidden states, avoiding an entire sequence-by-vocabulary
logit allocation. `step()` returns one token's logits and next state. Both use
the same layerwise causal scan and support gradients. Callers select eval mode
and `torch.no_grad()` for inference.

State is returned functionally. Layer activation checkpointing passes all state
tensors explicitly; recomputation never mutates caller-owned memory. Detached
physical caches have bounded storage, including contiguous slices that would
otherwise retain an entire segment's allocation. Model checkpoints include the
capacity and parameterization and reject incompatible architectures.

The reference supports FP32, absolute positions, FIFO, and aligned batch streams.
Padding, rebasing, retained prefixes and independent per-row resets are not
implemented. The pilot trainer uses one document stream at a time.

## Initialization

The library defaults to retention timescales `(512, 4096, 16384)` **eviction
updates**, initializing forget bias from `logit(exp(-1/tau))`. Input-gate bias
starts at -4. Write/recurrent weights are nonzero; read projections are nonzero.
Cell and exposed states start at exactly zero. No static learned value is used.

Initial score biases start at -20 only provisionally. Before training, calibrate
the **enabled** virtual branches on train-only sequences longer than the window.
The default targets total virtual mass `1e-4` at each layer/head's largest
observed mass. The first pass measures prospective virtual logits while reads
follow standard physical attention, so provisional biases cannot distort the
calibration activations. Further passes refine under enabled reads. A lower
quantile can be explicitly selected, but may allow disruptive outliers. Record mean, quantile and maximum
observed mass. This is near-function-preserving initialization, not exact parity
or a bound on unseen inputs. Zero memory values alone do not disable softmax mass.

## Document data and pilot training

The packed corpus used by the previous experiments has no explicit document
offsets. Do not silently carry memory across its independent blocks, and do not
infer boundaries from EOS token contents. Prepare source JSONL containing one
whole document per `{"text": ...}` record. In WikiText, reconstruct whole
articles first; treating each source row as a document can reset memory too often
to train long recall. Pin source provenance and the tokenizer/model revision.

```bash
python -m virtualsink.uptraining.recurrent_data \
  --train train-documents.jsonl --validation validation-documents.jsonl \
  --test test-documents.jsonl --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --source 'dataset/revision/license; whole-document reconstruction policy' \
  --output-dir recurrent-data
```

The resulting format records token checksums and explicit offset checksums.
The reader verifies boundaries, EOS endings, token IDs and counts. Every segment
has adjacent input and target tokens, including segment-boundary predictions;
it never crosses a document boundary.

The trainer is Slurm/CUDA-only. Start with a small profiling configuration:

```bash
python -m virtualsink.uptraining.train_recurrent \
  --model-path "$PRETRAINED_DIRECTORY" --data-dir recurrent-data \
  --output-dir recurrent-pilot --mode memory --variant memory \
  --capacity 64 --timescales 128 512 2048 \
  --segment-length 64 --unroll-segments 4 --steps 100 --warmup 10 \
  --checkpointing --max-hours 1
```

These smaller timescales/window are pilot settings, not the proposed final
512-token experiment. The graph spans four contiguous segments (up to 256 input
tokens per update). KV and memory both detach after each optimizer update;
forward memory survives. Direct gradient credit cannot cross a detach boundary.
To teach longer recall, increase the connected write-to-retrieval horizon and
add delayed-recall supervision; merely preserving detached state is insufficient.

Resume uses `--resume recurrent-pilot/last.pt` and otherwise identical arguments.
It restores model, AdamW, scheduler, detached stream state, document cursor and
RNG states, and checks source/config/corpus/pretrained file hashes. It never
silently repeats an exhausted corpus. SIGTERM/SIGUSR1 or the time budget produce
an optimizer-boundary checkpoint and `paused` summary; resubmission is explicit.

Validation reports token-weighted all/post-eviction/beyond-ten-window CE/PPL and
null/memory mass. If a region has no predictions, its CE/PPL is null. Validation
selects `best.pt` using post-eviction CE. At successful completion, the trainer
loads this selected checkpoint and reports its held-out test results with the
selected optimizer step. `last.pt` retains the final optimizer/cursor/state for
resume; it is saved before loading the selected model for testing.

## Controls and decision gates

Available trainer variants are memory, standard, null, write_disabled, and
forget_zero. For matched training controls, use `--mode attention` and the same
data/order/token budget/projection learning rate. The standard variant freezes
all memory parameters; null trains only its null score parameters. The memory
backend also supports zeroing memory values while retaining their logits, which
separates retrieved content from changes to attention normalization.

Do not apply the existing physical-mass anti-collapse penalty in the first
pilot: it would penalize useful reads from memory too. The pilot uses language
loss only; it does not impose equal slot utilization.

1. **Correctness:** disabled-virtual parity with native SWA; whole-segment versus
   segmented/tokenwise parity; causality; batch isolation/reset; first-eviction
   timing; closed-form cell retention; delayed gradients; checkpoint parity;
   bounded detached cache storage; document-boundary loss/resume equivalence.
2. **K80 smoke/profile:** `hpc/deploy_recurrent_smoke.py` freezes source in an
   isolated release and submits CPU numerical tests plus full pretrained
   Qwen1.5B calibration, 81-token/window-32 parity, and two memory-only AdamW
   steps. This uses existing packed tokens only as an arithmetic probe, with
   no memory-quality or document-training claim. Confirm logs after submission.
3. **Delayed recall:** facts before eviction, queries after eviction; random
   distractors; disjoint train/test examples and longer test delays. Compare
   disabled writes, erased cell history, zero read values and disabled reads.
   Require gains and intervention sensitivity before a long uptraining run.
4. **Matched streaming:** new document-preserving training controls, 512-token
   FIFO, retained/rebased prefix controls, static four sinks, multiple seeds,
   delayed recall and article loss. Count recurrent state and parameters in the
   memory budget. Compare equal physical cache and equal total state budgets
   separately. The earlier packed-training checkpoints are reference results,
   not matched training controls for this architecture.

The implementation is a tokenwise attention/recurrent scan within each layer.
QKV and MLP projections are batched over the segment, but attention and cell
updates remain sequential. K80 timing must precede a long-run budget; an eventual
segment-update or fused implementation requires its own parity and semantics
checks.

For Qwen1.5B in FP32, recurrent cells plus exposed states add about **0.98 MiB per
stream** (`2 * layers * query_heads * 3 * head_dim * 4` bytes). That equals the
physical K/V storage of 18 additional FIFO tokens for its 12 query heads/two KV
heads. An equal-state-budget comparison can therefore use memory cache 494
versus standard cache 512; equal-physical-cache comparisons use 512 for both.
The approximately 26.6 MiB of additional learned FP32 weights is a separate
model-memory cost, with additional gradient/optimizer costs during training.

## First verified GPU result — 13 October 2026

The original isolated gate 216951 failed its initialization mass check after the
16 numerical tests passed. That release and its logs are preserved. The repair
measures prospective scores under baseline attention first and targets maximum
observed mass rather than an interpolated lower quantile.

Replacement gate **216952 passed** on a Tesla K80 with PyTorch 1.12.1+cu113:

- 18 numerical/state/training/resume tests passed on the cluster.
- Qwen2.5-1.5B adds **6,982,976 parameters**.
- Maximum initial virtual mass: **0.000100171**, for target 0.0001.
- Segment/tokenwise maximum logit error: **0.000380516**; cell error 0.000010729.
- Both AdamW steps updated the write matrices in every layer. Every new
  parameter had a finite connected nonzero gradient on this probe.
- Memory-only steps over 81 tokens/window 32 took **13.23 and 13.20 seconds**.
- Peak reserved GPU memory: **7,168,065,536 bytes** (about 6.68 GiB).
- Slurm elapsed 1m33s; full-model probe elapsed 59.74s after numerical tests and
  model loading. These are small-window measurements, not a 512-window forecast.

Local reports: `artifacts/recurrent-sink-smoke-20261013-r2/smoke.json` and
`observation.json`. Collect further small reports with
`python hpc/collect_recurrent_smoke.py --attempt 2`.

`hpc/pilot_recurrent_recall.py` prepares the next controlled quality test: a
one-layer random model, cache eight, a fact followed by 24 distractors and a
query, 300 full-model updates each for memory and standard controls, held-out
delays 24/48/96, and write/read/value/forget ablations. Its one-layer physical
cache contains only projections of input embeddings, so it cannot relay the
first fact through contextual states once evicted. Chance accuracy is 12.5%.
This checks memory learning in a toy architecture; it does not establish gains
from inserting memory into pretrained Qwen. Submit an isolated pilot with
`python hpc/deploy_recurrent_smoke.py --kind recall`, then collect with
`python hpc/collect_recurrent_smoke.py --kind recall`.

## Controlled recall results

Pilot **216953** completed, but the original toy setup (memory bias -2, input
gate bias -4, 300 updates per arm) stayed at chance. At delay 24, both memory and
standard achieved 70/512 = 13.7%, and zeroing memory values did not change that
accuracy. Connected gradients were insufficient to make the toy model use memory.

Diagnostic **216954** then used memory bias +2, input gate bias -1, and 1,000
updates per arm. These changes increase early memory access and training time;
their effects are **not isolated**. The library/pretrained calibration defaults
were not changed to these toy values. Held-out results (512 examples each):

| Distractor tokens | Memory accuracy | Standard accuracy | Writes disabled | Forget gate forced zero | Memory values zeroed |
|---|---:|---:|---:|---:|---:|
| 24 (training delay) | 100.0% | 11.3% | 11.3% | 14.3% | 12.3% |
| 48 | 86.5% | 12.7% | 12.7% | 11.5% | 13.5% |
| 96 | 49.4% | 14.5% | 14.5% | 13.5% | 12.9% |

Memory training took 161.79 seconds; the matched standard arm took 66.59 seconds;
Slurm elapsed was 4m05s including tests/evaluation. The random model has 1,328 new
memory parameters. Cache capacity is eight, chance accuracy 12.5%, and all model
parameters were trained from scratch on the same ordered generated examples.

This demonstrates trainable storage/retrieval and sensitivity to the direct cell
retention gate on this task. It does not prove infinite retention or benefit in
pretrained Qwen. Longer-delay accuracy deteriorates, and at delay 96 the memory
model's CE is 3.065 versus standard CE 2.086: higher accuracy does not imply better
probability calibration. All results are one seed, with an adaptively selected
second diagnostic. Full reports are preserved under
`artifacts/recurrent-sink-recall-20261013-r1/recall.json` and
`artifacts/recurrent-sink-recall-20261013-r2/recall.json`.

Before long Qwen uptraining, isolate input/read-gate initialization and budget,
test a delayed-recall curriculum under the conservative calibrated read mass,
and then assemble the authoritative whole-document corpus and matched streaming
training controls. The independent packed-block trainer remains unchanged.
