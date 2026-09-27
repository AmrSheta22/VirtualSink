# Virtual sinks in from-scratch SmolLM training

## Experiment report — 15 September 2026

### Executive findings

Training with query-conditioned virtual sinks produced a clear improvement in the completed **full-attention, 512-token experiment**. After 30 updates, test cross-entropy fell from **7.690519 to 7.435304**, and perplexity fell from **2,187.51 to 1,694.77: a 22.52% reduction**. This is a positive training result worth preserving independently of the original streaming objective.

Disabling the virtual sinks after training, without retraining any weights, increased test cross-entropy only to **7.437833**. Approximately **99.01% of the cross-entropy advantage over the separately trained baseline survived removal**. The result supports further investigation of virtual sinks as a training intervention that may be dispensable at inference in this setting.

The completed **8,192-token / 512-window SWA experiment did not show a corresponding training advantage**: the paired test difference was consistent with zero. Both results belong in the conclusion; the evidence does not establish a benefit across attention regimes, training lengths, or seeds.

Neither completed baseline exhibited catastrophic cache-eviction failure in the tested streams. Therefore, these experiments did **not** demonstrate the originally intended mechanism—virtual sinks preventing a failure caused by eviction of physical attention sinks. They also do not rule out that mechanism in longer-trained models that actually develop such dependence.

### 1. Scope and experiment inventory

This report covers the four paired SmolLM from-scratch configurations found in the experiment directories, including stopped runs. Earlier pretrained-model uptraining experiments are outside scope. Baseline means a separately trained model with virtual sinks disabled; “sinks disabled” means an inference intervention on the sink-trained checkpoint. These are different controls.

| Experiment | Training attention | Sequence length | Sequences/update | Baseline / sink jobs | Outcome |
|---|---|---:|---:|---|---|
| Initial scratch, 10 September | Full causal | 2,048 | 512 | 214379 / 214380 | Cancelled; last logged updates 10 / 11 |
| SWA restart, 11 September | Sliding window 512, all layers | 8,192 | 128 | 214387 / 214388 | Both completed 30 updates |
| Full-attention restart, 13 September | Full causal | 8,192 | 128 | 214466 / 214467 | Cancelled before a completed update was logged |
| Short-context restart, 13 September | Full causal | 512 | 2,048 | 214473 / 214474 | Both completed 30 updates |

All planned pairs used 1,048,576 predicted tokens per optimizer update. Each completed model processed **31,457,280 training targets**. Because the compressed schedule assigns learning rate zero to update 1, only 29 updates had nonzero learning rate.

The initial 2,048-token pair used weak sink initialization (weight std 0.0001, bias −3.5). It was stopped when the experiment was redesigned to use SWA and stronger sink initialization. At the last common validation point, update 10, baseline CE was **7.841087**, versus **7.865427** for the sink model. There was no completed final test comparison; this partial result cannot substitute for one.

The full-attention 8,192-token pair used std 0.02 and bias +4. It was stopped after roughly 51 minutes to reduce training sequence length to 512. Its only logged validation was initialization: baseline **11.299098**, sink **11.299736**. It provides no trained-model outcome. The replacement pair began from fresh random weights, not those partial checkpoints.

### 2. Architecture, objective, and reproducibility

The model follows the SmolLM-135M configuration: 30 decoder layers, hidden width 576, MLP width 1,536, nine query heads and three KV heads per layer, tied token embeddings and output head, vocabulary 49,152, and RoPE theta 10,000. All shared weights were randomly initialized; pretrained model tensors were not loaded.

Each attention head has a learned virtual logit computed from its rotated query and a learned weight/bias. Its virtual value is zero. For fixed physical queries, keys, and values:

`head_output_with_sink = (1 − virtual_probability) × ordinary_physical_attention_output`

Thus the mechanism also provides a learned, query-dependent gate on the attention output. The equation describes the local operation for fixed activations, not equality between two complete models after their earlier layers have changed.

Both arms used model seed **8**, data seed **42**, and independently controlled sink parameter draws so that adding sinks did not shift shared parameter initialization. The shared-weight SHA256 was verified equal:

`6ae1a101ab6516fe007149ce9dfd7f0286ce77de0edc1d319e6f23f5d0af04d6`

| Setting | Completed experiments |
|---|---|
| Objective | Causal next-token cross-entropy only; auxiliary none, gamma 0 |
| Optimizer | AdamW, peak LR 0.003, betas 0.9/0.95, epsilon 1e−8, weight decay 0.01 |
| Gradient clipping | Global norm 1 |
| Precision / hardware | FP32, NVIDIA K80, PyTorch 1.12.1+cu113 |
| Microbatch | One sequence, accumulated to the shared token budget |
| Shared initialization | Normal std 0.0416; output/down projections scaled by 1/sqrt(60); RMSNorm weights 1 |
| Sink initialization | Weight std 0.02, bias +4 |
| Schedule | Compressed linear warmup, plateau, then 1−sqrt cooldown; first update LR 0 |
| Validation | Every update; 32,768 target tokens per validation evaluation |
| Checkpoints | Resumable last, best validation, and final; source/config/data hash checks |

The final update used LR **0.0003167184**; the schedule reaches zero after the final update rather than performing its last update at zero. Requeues resumed partial gradient accumulation and optimizer state; requeued Slurm elapsed times should not be mistaken for total training duration.

The recipe is a reconstruction of the public SmolLM recipe, not an exact reproduction of its original large-scale pretraining. FP32, a native memory-bounded implementation, shortened schedule, public corpus subsets and substitutions, and changed attention/context settings are explicit departures.

### 3. Data and evaluation samples

The target mixture was 70% FineWeb-Edu, 15% Cosmopedia, 8% Python-Edu, 6% OpenWebMath, and 1% StackOverflow, with rounded block quotas. The tokenizer was `HuggingFaceTB/cosmo2-tokenizer`, revision `4ce2318a3628e77279c939ed6a9f3f03034402de`.

Public data sources were pinned: `HuggingFaceTB/smollm-corpus`, `Avelina/python-edu-cleaned`, `open-web-math/open-web-math`, and `bigcode/stackoverflow-clean`. The Python source substitutes for unavailable text behind original blob identifiers. BigCode's cleaned StackOverflow substitutes for the unspecified original snapshot. These are recorded deviations, not claims of exact corpus replication.

SHA256 of stripped document text assigned documents to 98% training / 1% validation / 1% test. This prevents exact-text split overlap under that rule; it does not guarantee semantic deduplication. Documents were EOS-separated and packed without padding, and attention could cross document boundaries.

The SWA final test used **128 × 8,192** target tokens; the short full-attention test used **2,048 × 512**. Each totals 1,048,576 targets. Within each pair, evaluation targets were identical. Between experiments, packing, rounded quotas, context, and sample composition differ. Absolute CE across the two experiments is not a controlled test of attention architecture alone.

### 4. Completed training results

| Experiment / model | Final training CE | Final validation CE | Final test CE | Test perplexity |
|---|---:|---:|---:|---:|
| SWA baseline | 7.724734 | 7.557305 | 7.688645 | 2,183.41 |
| SWA virtual sink | 7.724519 | 7.560447 | 7.689196 | 2,184.62 |
| Full-512 baseline | 7.717180 | 7.552745 | 7.690519 | 2,187.51 |
| Full-512 virtual sink | 7.459171 | 7.280135 | 7.435304 | 1,694.77 |

Training CE averages microbatches before the corresponding optimizer update. Validation follows that update and uses a different fixed sample. Their difference is not itself evidence of faulty logging or leakage.

For SWA, baseline-minus-sink test CE was **−0.0005509**, with paired block-bootstrap 95% interval **[−0.0016899, +0.0006319]**. No advantage was established. The early trajectory was not uniformly smooth: sink validation CE rose from 9.8439 at update 2 to 11.1209 at update 3 before recovering to 8.4888 at update 4. Later losses converged toward the baseline.

For full-512, baseline-minus-sink test CE was **+0.2552148**, with paired block-bootstrap 95% interval approximately **[+0.2506, +0.2598]**. The sink model briefly trailed on validation at update 9, then gained an increasingly clear lead. Validation advantage grew to 0.1621 at update 22 and 0.2726 at update 30.

All completed training/evaluation jobs exited successfully. Full-512 final pre-clipping gradient norms were 0.2720 baseline and 0.2185 sink. Early baseline gradients were much larger, including 58.40 at update 3. Clipping prevented that norm from being the actual applied gradient norm. These observations support describing training as finite and successfully completed, but not as mature or optimally tuned pretraining: 30 updates and high final perplexity leave substantial learning unfinished.

Bootstrap intervals quantify variation across these test blocks, not training-seed uncertainty. Packed blocks can also share document context. There is one training seed per condition, so the positive result is established for this pair rather than demonstrated universally.

### 5. Removing sinks at inference

| Experiment | Sink-trained CE, enabled | Same weights, sinks disabled | Removal penalty, CE | Disabled perplexity |
|---|---:|---:|---:|---:|
| SWA | 7.68919623 | 7.68921402 | +0.00001779 | 2,184.66 |
| Full-512 | 7.43530433 | 7.43783271 | +0.00252838 | 1,699.06 |

The full-512 checkpoint retains a **0.25268645 CE** advantage over the separately trained baseline with sinks disabled. Its perplexity remains approximately **22.33% lower** than baseline. The fraction of the original CE advantage retained is:

`(7.690519159 − 7.437832706) / (7.690519159 − 7.435304329) = 99.01%`

This is the strongest evidence for a lasting training benefit. Removing the sink operation does not undo changes learned by the shared weights. Conversely, it is not evidence that active sinks make no difference: the measured full-512 penalty is small but positive. Nor is the intervention proof that every generation task, language, context length, or future checkpoint tolerates removal.

The tests disable sink contributions; they do not benchmark an exported sink-free implementation, inference latency, or deployment memory savings. Such savings require implementation and measurement. No retraining was performed for removal.

### 6. Virtual-sink concentration

Final supplemental attention diagnostics sampled eight test blocks and the latter half of each sequence. These statistics are narrower samples than the full test CE. They use the joint physical-plus-virtual probability denominator.

| Statistic, 270 heads | SWA | Full-512 |
|---|---:|---:|
| Mean virtual probability | 2.6246% | 5.9006% |
| Heads at least 99% virtual | 7 | 6 |
| Heads at least 30% virtual | 7 | 17 |
| Heads below 0.1% virtual | 254 | 213 |
| Share of total mean virtual mass in top seven heads | 98.7795% | 43.6967% |
| Effective number of active heads, (sum p)^2 / sum p² | 7.17 | 21.58 |

SWA usage was overwhelmingly seven saturated heads: zero-based L24H6, L25H0/H5/H6/H7, and L28H5/H7. Full-512 had six saturated heads: L19H0/H1/H2 and L27H3/H4/H5. Other full-512 heads used approximately 96%, 93%, 90%, 80%, 73%, and 68% virtual mass. The new distribution was concentrated but substantially broader.

The SWA validation history establishes that concentration emerged early: seven heads reached at least 99% at update 4; eleven did so at update 5; by update 14, seven heads carried 98.33% of the virtual mass. The final validation average of 2.6136% and top-seven share of 99.19% differ from final test statistics because the samples differ. Full-512 final validation mean was 6.478%, versus 5.901% in the test probe.

Early per-head statistics were retained even though periodic weight checkpoints were overwritten. The SWA sink model's best, last, and final checkpoints were at update 30; the baseline best checkpoint was update 25. Earlier distributions can be inspected, but arbitrary earlier weight interventions cannot be reconstructed from those logs alone.

Concentration is not sufficient evidence of functional importance. Saturated heads can have little effect on prediction if subsequent computation is insensitive to the reintroduced physical attention output.

### 7. Individual-head interventions in the SWA checkpoint

Job 214478 tested each of the seven saturated heads individually, all seven together, and all sinks together across 128 blocks / 1,048,576 targets. Disabling a sink means restoring physical-only attention normalization for that head; it does not remove the head itself.

| Disabled sink(s), zero-based | Mean CE change | 99th percentile token CE change | Largest token CE increase |
|---|---:|---:|---:|
| L24H6 | −0.00000761 | 0.003168 | 0.006384 |
| L25H0 | −0.00000372 | 0.002403 | 0.004443 |
| L25H5 | −0.00000706 | 0.003087 | 0.005724 |
| L25H6 | −0.00000393 | 0.002817 | 0.005146 |
| L25H7 | −0.00000398 | 0.002804 | 0.005098 |
| L28H5 | −0.00000109 | 0.003015 | 0.005301 |
| L28H7 | −0.00000105 | 0.002983 | 0.005422 |
| All seven | +0.00001583 | 0.019438 | 0.036120 |
| All sinks | +0.00001767 | 0.019480 | 0.054126 |

Every mean-change bootstrap interval included zero. No tested token incurred more than 0.1 CE harm. This directly weakens the hypothesis that large localized prediction damage was hidden by averaging in the old SWA model. The tiny difference between the all-sinks penalty here and the main evaluator reflects separate numerical evaluation paths; it does not alter the interpretation. These individual-head results apply to SWA; they were not repeated individually on full-512.

### 8. Physical attention sinks

Full-512 baseline attention to the first token averaged approximately **0.73%**, versus **0.21%** in the sink-trained model. First-four-token mass averaged **2.15% versus 0.49%**. For sampled positions 256–511, uniform physical attention would average approximately 0.27% per token. The baseline therefore showed first-token enrichment rather than no sink-like behavior.

The strongest baseline head, L19H2, assigned 7.76% on average to token zero and 23.53% to the first four tokens; those four held a majority on 19.8% of sampled queries. The strongest sink-model head assigned 4.01% to token zero. These are headwise means, not maxima over individual queries.

SWA absolute first-token mass was zero for the late queries by construction: token zero had left the window. Its relevant boundary measure was the oldest four visible tokens, averaging **0.9755% baseline versus 0.4809% sink** on the final test probe. Those values should not be confused with initial-prefix attention in full attention.

Physical initial-token preference was present, but dependence sufficient to cause eviction failure was not demonstrated. Preference, concentration, and causal necessity are distinct measurements.

### 9. Streaming and cache eviction

Completed diagnostics used two disjoint held-out packed FineWeb streams of 32,768 targets each, teacher-forced token by token. They are concatenated documents, not individual long books. Policies were full cache; FIFO capacity 512; four retained initial tokens plus 508 recent tokens; and FIFO with virtual sinks disabled for the sink-trained model. Capacity includes the current input token.

Absolute RoPE positions were preserved; there was no positional rebasing. Prefix retention is therefore a policy control, not an exact reproduction of StreamingLLM. FIFO first evicts at input position 512. For full-512, that is also the training-length boundary, so eviction and length extrapolation are not isolated. For SWA, positions beyond 8,192 are extrapolation, and unrestricted full-cache attention is a counterfactual mask rather than the trained mask.

Cached-versus-parallel parity passed for both policies and both models. Maximum absolute logit discrepancies were below 0.000009 for SWA and below 0.000178 for full-512, against tolerance 0.003. These checks support cache implementation consistency, not the scientific hypothesis by themselves.

For the SWA pair, neither baseline nor sink model collapsed under FIFO. Full-cache, FIFO, prefix retention, and sink removal differed only slightly in mean loss. This is consistent with training already exposing the model to predictions well beyond its 512-token window.

For full-512, mean CE after position 512 was:

| Model / policy | Stream 0 | Stream 1 |
|---|---:|---:|
| Baseline, full cache | 7.688435 | 7.501960 |
| Baseline, FIFO | 7.688354 | 7.501874 |
| Sink model, full cache | 7.791083 | 7.599511 |
| Sink model, FIFO | 7.456885 | 7.237024 |
| Sink model, prefix four + recent | 7.457755 | 7.270077 |
| Sink model, FIFO, sinks disabled | 7.457858 | 7.237517 |

The sink-trained model retained its advantage under FIFO even when sinks were disabled. Baseline eviction incurred negligible mean cost. In contrast, unrestricted context hurt the sink-trained model: FIFO improved its beyond-training-length CE by **0.3342 and 0.3625**. The sink model's unrestricted-cache loss was also worse than baseline's unrestricted-cache loss on these streams. The short-context training benefit therefore does not automatically extend to unrestricted long-context inference.

This result is compatible with an extrapolation problem and does not establish its cause. Mean full-stream excess losses differ slightly from the above beyond-length differences because they include the initial 512 targets. Prefix retention did not improve on FIFO here and was notably worse for the sink model on stream 1.

The original predicted signature—baseline collapses after eviction while virtual sinks prevent collapse—was absent. Perplexity greater than 1,000 alone is not a failure criterion, since these immature models already had that scale of perplexity before eviction.

### 10. Interpretation and final conclusions

**Supported finding: training benefit in full-512.** The matched full-attention experiment demonstrated better held-out predictions after training with virtual sinks. That result should be retained as a substantive positive outcome, even though it differs from the original objective.

**Supported finding: most of that benefit survives inference removal.** Roughly 99% of the CE advantage persisted without active sinks. The beneficial difference is largely retained in the learned shared weights, rather than requiring the sink operation at every inference step in these evaluations.

**Supported boundary: no benefit established in SWA.** The completed SWA result was effectively a tie. Changing sequence length, attention, block packing, and sample composition together prevents assigning the contrast solely to full versus sliding-window attention.

**Unproven mechanism: improved optimization through gating.** A zero-valued sink scales physical attention output and changes gradient flow. It could suppress unhelpful early head outputs and alter the learning trajectory. Early gradient differences and retention after removal are consistent with this account, but no fixed-gate control or causal training ablation establishes it. The result should not be presented as proven regularization, proven architectural superiority, or proven immunity to physical-sink eviction.

**Open question: insufficient training for eviction dependence.** Thirty updates may be too few to develop strong reliance on physical sinks. Some initial-token concentration emerged, but the baseline did not suffer from eviction. Longer training might change that; it is not guaranteed. Weak context use is another possible explanation. Neither hypothesis was isolated here.

**Practical research conclusion.** Preserve the full-512 result as evidence for “train with virtual sinks, optionally remove them at inference,” qualified by the tested data, seed, and inference policy. Preserve the original streaming hypothesis as unresolved rather than validated or decisively rejected.

### 11. Follow-up experiments justified by these results

1. Repeat the paired full-512 training across several seeds and evaluate document-level or suitably grouped uncertainty, in addition to block bootstrap.
2. Continue paired training longer, saving milestone checkpoints and tracking context dependence, first-token enrichment, and eviction penalties together.
3. Compare query-conditioned sinks with fixed attention scaling and learned scalar head gates to isolate whether adaptive gating explains the gain.
4. Test train-with-sinks/remove-at-inference across domains and generation tasks, and benchmark an actually stripped inference implementation.
5. Separate eviction from positional extrapolation by training beyond the evaluation window and comparing cache policies on identical targets within the trained positional range.

These are recommendations, not newly submitted jobs. The most important prerequisite for the original objective is a baseline with a reproducible eviction penalty.

### 12. Evidence and reproducibility record

The report was assembled from saved run configurations, metrics, final summaries, comparison reports, diagnostic reports, and Slurm accounting. The local evidence snapshot intentionally omits large per-block and physical-gate arrays; authoritative originals remain on HPC. Automatically generated interpretation strings were not accepted as proof of significance merely because a numerical delta was positive.

HPC roots, relative to `/cluster/users/alex116u1/data/`:

- `smollm-scratch-20260910`: cancelled full-2,048 pair.
- `smollm-swa8192-20260911`: completed SWA pair; comparison 214389; individual-head job 214478.
- `smollm-full8192-20260913`: cancelled full-8,192 pair; completed old-SWA diagnostics 214465 in `diagnostics-swa`.
- `smollm-full512-20260913`: completed full-512 pair; comparison 214475; diagnostics 214476; concentration monitor 214479.

The SWA manifest SHA256 was `75f189d00b37351c74b1cdd9da417237d23880269fe2c0e2de1effef3bbe1df8`; full-512 was `0d8cdb9cb8a074864e481bdec43c5e33f04f69bb2ca01fc0f0224f4baa121cdb`. Complete source-file hashes are in the saved configurations. Separate rebased diagnostic artifacts found on HPC are outside this report's absolute-position evaluation protocol and were not used to infer additional training runs or outcomes.

Local supporting files:

- [Evidence snapshot](../../artifacts/scratch-report/evidence.json)
- [Accounting and data provenance](../../artifacts/scratch-report/provenance.txt)
- [Report builder](../../artifacts/scratch-report/build_report.py)
- [Initial recipe documentation](../smollm-pretraining.md)
- [SWA protocol](../smollm-swa8192.md)
- [Full-attention streaming protocol](../full-attention-streaming-experiment.md)
- [Full-512 restart](../full512-restart.md)

The public-recipe origin and pinned reference are recorded in those protocol files and in each saved run configuration. This report's numerical claims come from experiment artifacts, not external performance claims.

### Appendix A. Complete validation trajectories

Each row compares equal completed update counts within its pair. Positive differences favor virtual sinks.

#### SWA 8,192 / window 512

| Update | Baseline CE | Sink CE | Baseline minus sink |
|---:|---:|---:|---:|
| 0 | 11.289237 | 11.290711 | -0.001474 |
| 1 | 11.289237 | 11.290711 | -0.001474 |
| 2 | 9.926900 | 9.843868 | +0.083032 |
| 3 | 10.299536 | 11.120910 | -0.821374 |
| 4 | 8.695372 | 8.488786 | +0.206586 |
| 5 | 8.200575 | 8.066812 | +0.133763 |
| 6 | 7.874211 | 7.860611 | +0.013600 |
| 7 | 7.779353 | 7.779576 | -0.000223 |
| 8 | 7.733849 | 7.763205 | -0.029356 |
| 9 | 7.722049 | 7.762093 | -0.040044 |
| 10 | 7.717569 | 7.723918 | -0.006349 |
| 11 | 7.678022 | 7.672876 | +0.005145 |
| 12 | 7.657241 | 7.650713 | +0.006528 |
| 13 | 7.659841 | 7.665630 | -0.005789 |
| 14 | 7.647927 | 7.651452 | -0.003525 |
| 15 | 7.623698 | 7.623900 | -0.000202 |
| 16 | 7.606095 | 7.605666 | +0.000429 |
| 17 | 7.606393 | 7.603302 | +0.003091 |
| 18 | 7.619075 | 7.612061 | +0.007014 |
| 19 | 7.628223 | 7.618993 | +0.009230 |
| 20 | 7.649630 | 7.640089 | +0.009541 |
| 21 | 7.651892 | 7.643954 | +0.007938 |
| 22 | 7.627793 | 7.623984 | +0.003809 |
| 23 | 7.592514 | 7.593308 | -0.000794 |
| 24 | 7.568885 | 7.573208 | -0.004324 |
| 25 | 7.554289 | 7.562194 | -0.007905 |
| 26 | 7.557007 | 7.565280 | -0.008273 |
| 27 | 7.561587 | 7.567524 | -0.005937 |
| 28 | 7.560473 | 7.564585 | -0.004112 |
| 29 | 7.558858 | 7.562197 | -0.003339 |
| 30 | 7.557305 | 7.560447 | -0.003142 |

#### Full attention 512

| Update | Baseline CE | Sink CE | Baseline minus sink |
|---:|---:|---:|---:|
| 0 | 11.311271 | 11.292500 | +0.018772 |
| 1 | 11.311271 | 11.292500 | +0.018772 |
| 2 | 10.411784 | 9.736325 | +0.675459 |
| 3 | 10.231773 | 9.228688 | +1.003085 |
| 4 | 8.879998 | 8.207417 | +0.672581 |
| 5 | 8.234246 | 7.858653 | +0.375593 |
| 6 | 7.906993 | 7.726130 | +0.180863 |
| 7 | 7.812986 | 7.734338 | +0.078648 |
| 8 | 7.754158 | 7.711364 | +0.042794 |
| 9 | 7.720013 | 7.728834 | -0.008820 |
| 10 | 7.740289 | 7.731172 | +0.009118 |
| 11 | 7.728279 | 7.694090 | +0.034188 |
| 12 | 7.700969 | 7.642546 | +0.058423 |
| 13 | 7.651933 | 7.610808 | +0.041125 |
| 14 | 7.641217 | 7.578853 | +0.062364 |
| 15 | 7.649687 | 7.565773 | +0.083914 |
| 16 | 7.644151 | 7.551361 | +0.092790 |
| 17 | 7.632245 | 7.545222 | +0.087023 |
| 18 | 7.626056 | 7.540731 | +0.085325 |
| 19 | 7.625265 | 7.529198 | +0.096067 |
| 20 | 7.622022 | 7.504675 | +0.117347 |
| 21 | 7.618567 | 7.475952 | +0.142616 |
| 22 | 7.611596 | 7.449481 | +0.162115 |
| 23 | 7.601257 | 7.425998 | +0.175259 |
| 24 | 7.586355 | 7.395174 | +0.191181 |
| 25 | 7.575853 | 7.360869 | +0.214984 |
| 26 | 7.569878 | 7.330872 | +0.239006 |
| 27 | 7.566625 | 7.312809 | +0.253816 |
| 28 | 7.561184 | 7.297553 | +0.263632 |
| 29 | 7.556070 | 7.286248 | +0.269822 |
| 30 | 7.552745 | 7.280135 | +0.272611 |

### Appendix B. Original streaming summaries

These machine-generated summaries include all policies and distinguish full-stream excess CE from within/beyond-training-length CE.

#### Old SWA checkpoints



Teacher-forced held-out packed FineWeb streams. Input positions are zero-based; FIFO first evicts at input position 512. Beyond 8192 is length extrapolation.

Loss increases are measured against the same model's full-cache reference on identical target tokens. Neither flat perplexity nor baseline collapse is assumed.

| Variant | Stream | Policy | CE within 8k | CE beyond 8k | Mean excess CE vs full cache |
|---|---:|---|---:|---:|---:|
| query_sink | 0 | fifo | 7.761554 | 7.675972118207331 | -0.0005764317611465231 |
| query_sink | 0 | fifo_disabled | 7.761247 | 7.676376748868885 | -0.0003496726058074273 |
| query_sink | 0 | full | 7.761296 | 7.676826676295605 | 0.0 |
| query_sink | 0 | prefix4 | 7.761451 | 7.676235214380237 | -0.00040490498213330284 |
| query_sink | 1 | fifo | 7.300950 | 7.584762407098121 | -0.00015039979189168662 |
| query_sink | 1 | fifo_disabled | 7.300685 | 7.584438465069979 | -0.00045973562373546883 |
| query_sink | 1 | full | 7.301002 | 7.584945662344883 | 0.0 |
| query_sink | 1 | prefix4 | 7.300790 | 7.584553409707344 | -0.0003473455217317678 |
| standard | 0 | fifo | 7.761364 | 7.672267990245018 | -0.00023828811390558258 |
| standard | 0 | full | 7.761451 | 7.6725566722355625 | 0.0 |
| standard | 0 | prefix4 | 7.761364 | 7.672269074605235 | -0.00023738886375213042 |
| standard | 1 | fifo | 7.308997 | 7.582727046402094 | -0.000301790569210425 |
| standard | 1 | full | 7.309164 | 7.58307368157936 | 0.0 |
| standard | 1 | prefix4 | 7.309005 | 7.582733132566015 | -0.00029519425879698247 |
#### Full-512 checkpoints



Teacher-forced held-out packed FineWeb streams. Input positions are zero-based; FIFO first evicts at input position 512. Training length is recorded per result; positions beyond it test extrapolation.

Loss increases are measured against the same model's full-cache reference on identical target tokens. Neither flat perplexity nor baseline collapse is assumed.

| Variant | Stream | Policy | CE within training length | CE beyond training length | Mean excess CE vs full cache |
|---|---:|---|---:|---:|---:|
| query_sink | 0 | fifo | 7.400028 | 7.456885269418033 | -0.32897621460142545 |
| query_sink | 0 | fifo_disabled | 7.398750 | 7.457857647611361 | -0.32803899044301943 |
| query_sink | 0 | full | 7.400028 | 7.791083328695672 | 0.0 |
| query_sink | 0 | prefix4 | 7.400028 | 7.457754951994127 | -0.3281201208155835 |
| query_sink | 1 | fifo | 7.278409 | 7.237024061987916 | -0.3568233552214224 |
| query_sink | 1 | fifo_disabled | 7.280708 | 7.237517437284132 | -0.35630177394341445 |
| query_sink | 1 | full | 7.278409 | 7.59951127999063 | 0.0 |
| query_sink | 1 | prefix4 | 7.278409 | 7.27007676071177 | -0.324287104915129 |
| standard | 0 | fifo | 7.605791 | 7.688354039202547 | -7.95009036664851e-05 |
| standard | 0 | full | 7.605791 | 7.68843480202532 | 0.0 |
| standard | 0 | prefix4 | 7.605791 | 7.688353121672417 | -8.040409738896415e-05 |
| standard | 1 | fifo | 7.476471 | 7.501873605256338 | -8.475573122268543e-05 |
| standard | 1 | full | 7.476471 | 7.501959706316629 | 0.0 |
| standard | 1 | prefix4 | 7.476471 | 7.501875960245906 | -8.243753836723045e-05 |