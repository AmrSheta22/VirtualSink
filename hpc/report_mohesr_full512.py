"""Read-only compact report; run on manage2 with the experiment root argument."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
jobs = (root / "jobs.tsv").read_text().splitlines()
ids = [line.split()[1].split(";")[0] for line in jobs]
report = {"jobs": jobs}
for name, command in {
    "queue": ["squeue", "-u", str(os.getuid()), "-o", "%.24i %.20j %.12T %.10M %R"],
    "accounting": ["sacct", "-j", ",".join(ids), "--format=JobID,JobName,State,ExitCode,Elapsed", "-P"],
}.items():
    result = subprocess.run(command, capture_output=True, text=True)
    report[name] = result.stdout if result.returncode == 0 else result.stderr
report["arms"] = []
for seed in (8, 9, 10):
    for variant in ("standard", "query_sink"):
        run = root / f"seeds/seed{seed}/runs/{variant}"
        row = dict(seed=seed, variant=variant)
        for name in ("config", "status", "summary"):
            path = run / (name + ".json")
            if path.exists():
                value = json.loads(path.read_text())
                if name == "config":
                    row["shared_initial_weights_sha256"] = value["shared_initial_weights_sha256"]
                elif name == "summary":
                    row["final_test"] = {k: value["final_test"][k] for k in ("ce", "perplexity")}
                    row["disabled_minus_enabled_ce"] = value.get("disabled_minus_enabled_ce")
                else:
                    row[name] = value
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            # Ignore only an incomplete final write while the trainer is active.
            for line in metrics.read_text().splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["latest"] = {k: v for k, v in value.items() if k not in ("validation", "attention")}
                if "micro_seconds" in value:
                    row["training_progress"] = row["latest"]
                if "lm_ce" in value:
                    row["last_optimizer_update"] = row["latest"]
                if "validation" in value:
                    row["validation"] = {k: value["validation"][k] for k in ("ce", "perplexity")}
                    row["validation_step"] = value["step"]
        report["arms"].append(row)
    comparison = root / f"seeds/seed{seed}/comparison/comparison.json"
    if comparison.exists():
        value = json.loads(comparison.read_text())
        report[f"comparison_seed{seed}"] = {k: value[k] for k in ("test_comparison", "disabled_intervention")}
for variant in ("standard", "query_sink"):
    benchmark = root / ("benchmark-" + variant + ".json")
    if benchmark.exists():
        report["benchmark_" + variant] = json.loads(benchmark.read_text())
for name in ("microbatch-benchmark.json", "selected-microbatch.txt"):
    path = root / name
    if path.exists():
        report[name] = json.loads(path.read_text())
aggregate = root / "aggregate.json"
if aggregate.exists():
    report["aggregate"] = json.loads(aggregate.read_text())
report["errors"] = {path.name: path.read_text()[-4000:] for path in (root / "logs").glob("*.err") if path.stat().st_size}
print(json.dumps(report, indent=2))
