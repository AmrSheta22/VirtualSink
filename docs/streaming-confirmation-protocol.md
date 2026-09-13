# Streaming confirmation protocol (fixed before new results)

Primary scope: completed regularized step1000 Qwen2.5-0.5B and SmolLM-135M
SWA uptraining checkpoint pairs. Same native evaluation and absolute RoPE policy
as release2214820; no checkpoint, window, initialization or training changes.

Population: all official WikiText-103 test articles with at least8193 tokens
under both pinned model tokenizers. There are14 eligible articles out of62.
Exclude the two already inspected exploratory articles (American Beauty and
2003 Pacific typhoon season), leaving12 new articles. Titles are fixed in
CONFIRMATION_TITLES; no performance-based selection. Results generalize only
within this narrow long-WikiText population, not books or all streaming tasks.

Each new article contributes8192 teacher-forced predictions under four policies:
standard FIFO512; standard4 initial+508 recent; trained virtual-sink FIFO512;
same sink checkpoint FIFO512 with virtual sinks disabled. Total786432 predictions
across12 articles,2 models,4 policies. Physical cache includes current token.

Primary endpoint per model: article-average standard-FIFO-minus-virtual-FIFO
cross-entropy at input positions512–8191. Positive means virtual sinks help.
Report mean, document-paired bootstrap95% interval (10000 resamples, seed1729),
PPL ratio and transformed interval, article wins/ties and worst article. Articles
receive equal weight and have equal scored lengths. Correlated topics may limit
independence. Intervals do not cover training-seed variability. Models are
reported separately, not pooled. Original two articles are excluded from the
confirmation statistics.

Secondary descriptive endpoints: before eviction, beyond10w, disabled-sink
intervention, retained-prefix control, mean virtual mass. No claim of multiple-
comparison-corrected significance for these endpoints. Absolute-position prefix
retention is not exact StreamingLLM; no full-cache reference or RoPE rebasing
is introduced. Higher sink use alone is not success.

All per-article jobs verify real-checkpoint cached-vs-parallel logits across
eviction before scoring. Results and progress are saved per policy/article.
Two model arrays of12 jobs each are capped at2 simultaneous jobs per array
(4 GPUs maximum total); each unit has2h limit. Previous measured throughput
suggests roughly14–16 GPU-hours total, or about4h with four GPUs continuously
available, plus queue time. No automatic retries or resubmission.

A CPU report job runs after both arrays terminate. It requires complete results
for all24 model/article units; otherwise it writes INCOMPLETE.json and explicitly
withholds a final conclusion. Complete reports include machine-readable
article_results.json, statistics.json and REPORT.md. No recurring monitor.
