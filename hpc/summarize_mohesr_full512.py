"""Aggregate independent seed comparisons; do not pool blocks across seeds."""
import json
import os
from pathlib import Path
import statistics

root = Path(os.environ["MOHESR_EXPERIMENT"])
rows = []
for seed in (8, 9, 10):
    report = json.loads((root / f"seeds/seed{seed}/comparison/comparison.json").read_text())
    test = report["test_comparison"]
    rows.append(dict(seed=seed, **test, disabled_intervention=report["disabled_intervention"]))
gains = [r["ce_reduction"] for r in rows]
result = dict(seeds=rows, mean_ce_reduction=statistics.mean(gains),
              sd_ce_reduction=statistics.stdev(gains),
              positive_seeds=sum(x > 0 for x in gains),
              note="Three independent initialization seeds on one fixed dataset; block bootstrap intervals are within-seed only.")
(root / "aggregate.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
