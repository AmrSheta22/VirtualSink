# Qwen0.5B full held-out article evaluation

Experiment `qwen05-full-article-evaluation-20261017-r1` corrects the previous
4096-prediction final evaluation cap. It performs inference only on the fixed,
previously validation-selected step2048 checkpoints. All57 test and57 validation
articles are evaluated completely, longest first, resetting memory only between
articles. Physical capacity128, segment256, FP32, original optimized implementation.

Jobs217343 (memory),217344 (same weights with all virtual reads/null/writes
disabled),217347 (trained FIFO) run independently. FIFO217345 was intentionally
cancelled before any startup output or scores after a comp004 launch stall;
replacement217346 also stalled before batch output on comp005 and was stopped.
FIFO217347 runs the unchanged frozen script on the second GPU of the already
active comp002 node. Both replacements preserved all source/checkpoint bytes
and discarded no evaluation scores. Teacher remains stopped.
The frozen source archive and hard links to the unchanged checkpoints live in
`/cluster/users/alex116u1/data/qwen05-full-article-evaluation-20261017-r1`.

Test coverage:293294 predictions, longest article16350 predictions.
The four position bands0–4096/4096–8192/8192–12288/12288–16384 contain
57/27/15/6 articles and173315/75659/36030/8290 scored tokens respectively.
Validation coverage:234496 predictions, longest14127; corresponding counts
57/24/5/2 articles and174532/47475/9732/2757 tokens.
These are zero-based input positions predicting the next token.

Use `hpc/collect_qwen_full_articles.py` for read-only collection of small logs
and per-segment loss sums; checkpoints are never downloaded. Then use
`hpc/analyze_qwen_full_articles.py` for matched-position token-weighted CE and
per-article paired differences. Partial reports compare only segments present
in all three arms. Complete budgets are asserted against `data-coverage.json`.
Artifacts are under `artifacts/qwen05-full-article-evaluation-20261017-r1`.

This is a post-hoc extension of an already inspected test split, without new
checkpoint selection. Different bands contain different articles, so raw loss
across positions is confounded by article difficulty. Deep absolute position
does not establish the age of useful information. Six long test articles and
two long validation articles limit the deepest band's generality. No claims
of distant associative fact retrieval follow from continuation loss alone.
