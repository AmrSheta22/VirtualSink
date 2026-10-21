# Pythia prefix-replacement results

Collected 18 October 2026. Recovered validation 217355_2, all 24 evaluation
tasks 217356, and aggregation 217357 completed successfully on 17 October.
All 1,572,864 predictions are present. The old node-failed validation remains
historical; it did not supply results. Matched baseline policies passed the
required reproduction check against previously saved per-token losses.

## Conclusion

The tested trained zero-value virtual sinks do not replace the retained prefix.
Unlike the SmolLM screen, these checkpoints show a large prefix benefit under
controlled cache-relative positioning. Enabling either trained sink fails to
remove that dependence. This is evidence about these checkpoints and training
recipes, not an impossibility result for virtual sinks generally.

Post-eviction perplexity below covers the same prediction indices 1024–65535
for every policy. Both columns use cache-relative RoPE, isolating retention
from positional handling. Lower is better.

| Checkpoint / sink state | FIFO | Retain first four |
|---|---:|---:|
| Untouched pretrained | 67.987 | 12.049 |
| Standard uptrained | 81.725 | 13.434 |
| Legacy sink enabled | 227.024 | 16.373 |
| Legacy sink disabled | 140.009 | 17.665 |
| Calibrated sink enabled | 81.361 | 13.454 |
| Calibrated sink disabled | 83.233 | 13.460 |

The calibrated sink reduces FIFO CE by 0.022749, approximately 2.25% lower
perplexity, but leaves a 1.799628 CE prefix rescue. With the sink disabled,
prefix rescue is 1.821919 CE. Thus only a small part of prefix dependence is
reduced. Enabled FIFO remains approximately six times the perplexity of enabled
prefix retention.

The legacy sink worsens FIFO CE by 0.483351 when enabled, approximately 62.1%
higher perplexity. Its prefix rescue increases from 2.070142 CE disabled to
2.629418 CE enabled. It can help the retained-prefix policy modestly while
failing badly as a substitute for that prefix.

## Position handling matters

Simply retaining four tokens at their absolute positions is harmful in several
arms: standard-uptrained perplexity is 208.980, compared with FIFO 81.741 and
retained-prefix/cache-relative 13.434. Untouched pretrained values are 185.769,
67.996 and 12.049 respectively. Therefore the original bundled StreamingLLM
benefit cannot be attributed to retaining tokens alone with arbitrary positions.
The controlled relative-versus-relative comparison above still gives a large
prefix effect, so the virtual-sink replacement failure persists after accounting
for this confound.

## Interpretation and limits

These sinks were trained on intact full-causal 512-token blocks, not with prefix
eviction. The results show that this recipe did not produce functional prefix
replacement. They do not isolate whether the limiting cause is the objective,
short uptraining, zero values, limited score expressivity, or their interaction.
No claim is made that adding learned values or eviction-aware training must work.

One training seed and the previously examined first 65,536 PG19 predictions
were used. Cache persists across separately tokenized books; there are no
cross-book scored targets. This is a controlled diagnostic, not independent
confirmation across books or training seeds. Untouched pretrained performs
better than the uptrained standard on this screen, another reason not to
interpret these pilot recipes as mature language-model improvements.

Protocol: docs/pythia-positive-control-20261017.md.
Full measurements: artifacts/pythia-positive-control-20261017/aggregate.json.
