"""Fetch completed run evidence read-only and summarize all six seed pairs."""
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from bibalex_remote import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/smollm-seed-analysis-20261016"
EXPERIMENTS = {
    "short": ("/cluster/users/alex116u1/data/smollm-full512-replication-20261009", (8, 9, 10)),
    "warm70": ("/cluster/users/alex116u1/data/smollm-warm70-20261014", (11, 12, 13)),
}


def fetch():
    client = connect()
    try:
        sftp = client.open_sftp()
        for exp, (remote, seeds) in EXPERIMENTS.items():
            for seed in seeds:
                for arm in ("standard", "query_sink"):
                    for filename in ("config.json", "summary.json", "metrics.jsonl"):
                        local = OUT / "raw" / exp / f"seed{seed}" / arm / filename
                        local.parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(f"{remote}/seeds/seed{seed}/runs/{arm}/{filename}", str(local))
                local = OUT / "raw" / exp / f"seed{seed}" / "comparison.json"
                sftp.get(f"{remote}/seeds/seed{seed}/comparison/comparison.json", str(local))
                print(f"Fetched {exp} seed {seed}", flush=True)
            sftp.get(remote + "/data/manifest.json", str(OUT / "raw" / exp / "manifest.json"))
    finally:
        client.close()


def read(exp, seed, arm, filename):
    path = OUT / "raw" / exp / f"seed{seed}" / arm / filename
    if filename.endswith("jsonl"):
        return [json.loads(s) for s in path.read_text().splitlines()]
    return json.loads(path.read_text())


def stats(values):
    a = np.array(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()), max=float(a.max()))


def analyze():
    result = []
    curves = []
    for exp, (_, seeds) in EXPERIMENTS.items():
        for seed in seeds:
            comparison = json.loads((OUT / "raw" / exp / f"seed{seed}/comparison.json").read_text())
            row = dict(experiment=exp, seed=seed, **comparison["test_comparison"])
            intervention = comparison["disabled_intervention"]
            row["disabled_delta"] = intervention["disabled_minus_enabled_ce"]
            row["gain_retained_when_disabled"] = ((row["ce_reduction"] - row["disabled_delta"]) / row["ce_reduction"])
            row["ppl_reduction_pct"] = 100 * (1 - math.exp(-row["ce_reduction"]))
            row["arms"] = {}
            configs = []
            for arm in ("standard", "query_sink"):
                config = read(exp, seed, arm, "config.json")
                configs.append(config)
                summary = read(exp, seed, arm, "summary.json")
                metrics = read(exp, seed, arm, "metrics.jsonl")
                # Requeues can repeat initial/evaluation records: keep last row per step.
                updates = {r["step"]: r for r in metrics if "lm_ce" in r}
                evaluations = {r["step"]: r for r in metrics if "validation" in r}
                rates = sorted(updates)
                gate = np.array(summary["final_test"]["physical_gate_by_layer_head"])
                late = np.array(summary["attention"]["by_layer_head"])
                arm_stats = dict(final_train_ce=updates[max(updates)]["lm_ce"],
                    max_grad_norm=max(r["grad_norm"] for r in updates.values()),
                    clip_fraction=sum(r["grad_norm"] > 1 for r in updates.values()) / len(updates),
                    final_sink_grad_norm=updates[max(updates)]["sink_grad_norm"],
                    last5_sink_grad_mean=statistics.mean(updates[s]["sink_grad_norm"] for s in rates[-5:]),
                    all_position_sink_mass=float((1-gate).mean()),
                    late_sink_mass=float(late[:, :, 2].mean()),
                    saturated_virtual_heads=int((gate < .01).sum()),
                    saturated_physical_heads=int((gate > .99).sum()),
                    intermediate_heads=int(((gate >= .01) & (gate <= .99)).sum()),
                    sink_mass_by_layer=(1-gate).mean(1).tolist(),
                    late_first4_mass=float(late[:, :, 1].mean()),
                    thresholds={str(t): next((s for s in sorted(evaluations) if evaluations[s]["validation"]["ce"] < t), None)
                                for t in (8., 7.6, 7.4, 7.2, 7., 6.8)},
                    per_block_ce_quantiles=np.quantile(np.array(summary["final_test"]["per_block_ce"]), [.05,.5,.95]).tolist())
                row["arms"][arm] = arm_stats
                for step, r in sorted(evaluations.items()):
                    g = np.array(r["validation"]["physical_gate_by_layer_head"])
                    l = np.array(r["attention"]["by_layer_head"])
                    curves.append(dict(experiment=exp, seed=seed, arm=arm, step=step,
                        validation_ce=r["validation"]["ce"],
                        sink_mass=float((1-g).mean()), late_sink_mass=float(l[:, :, 2].mean()),
                        saturated_virtual_heads=int((g < .01).sum()),
                        intermediate_heads=int(((g >= .01) & (g <= .99)).sum()),
                        grad_norm=updates.get(step, {}).get("grad_norm"),
                        sink_grad_norm=updates.get(step, {}).get("sink_grad_norm")))
            ignored = {"variant", "output_dir", "data_dir", "save_seconds", "max_hours"}
            row["pair_config_differences"] = [k for k in configs[0] if k not in ignored and configs[0][k] != configs[1][k]]
            a = np.array(read(exp, seed, "standard", "summary.json")["final_test"]["per_block_ce"])
            b = np.array(read(exp, seed, "query_sink", "summary.json")["final_test"]["per_block_ce"])
            row["blocks_improved_fraction"] = float((a > b).mean())
            row["block_ce_reduction_quantiles"] = np.quantile(a-b, [.05,.5,.95]).tolist()
            row["gate_statistics"] = comparison["gate_statistics"]
            result.append(row)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    (OUT / "curves.json").write_text(json.dumps(curves, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    analyze()
