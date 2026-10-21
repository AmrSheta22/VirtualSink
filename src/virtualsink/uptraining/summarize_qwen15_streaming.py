"""Recompute article scores from token losses, require all units, bootstrap articles."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .evaluate_streaming import CONFIRMATION_TITLES, write
from .qwen15_streaming import POLICIES
from .summarize_streaming_confirmation import paired_effect

REGIONS = {"pre_eviction": (0, 512), "post_eviction": (512, 8192), "beyond_10w": (5120, 8192)}


def token_scores(result):
    values = np.asarray(result["per_token_ce"], dtype=float)
    if values.shape != (8192,) or not np.isfinite(values).all():
        raise ValueError("8192 finite per-token losses required")
    return {name: float(values[left:right].mean()) for name, (left, right) in REGIONS.items()}


def summarize(root):
    rows, missing, common = [], [], None
    expected = json.loads((root / "checkpoint-identity.json").read_text())["checkpoint_sha256"]
    for index, title in enumerate(CONFIRMATION_TITLES):
        folder = root / "results" / f"article{index:02d}"
        if not (folder / "complete.json").exists():
            missing.append(str(folder))
            continue
        config = json.loads((folder / "config.json").read_text())
        assert config["article_index"] == index and config["article"]["title"] == title
        assert config["best_step"] == 975 and config["checkpoint_sha256"] == expected
        assert config["capacity"] == 512 and config["predictions"] == 8192 and config["policies"] == POLICIES
        cohort = {k: config[k] for k in ["training_config", "checkpoint_sha256", "best_step", "capacity", "predictions", "policies", "source_sha256"]}
        if common is None:
            common = cohort
        assert cohort == common, "inconsistent source/checkpoint settings"
        input_ids = None
        for policy in POLICIES:
            result = json.loads((folder / (policy + ".json")).read_text())
            assert result["capacity"] == result["peak_cache_entries"] == 512
            assert result["prefix"] == (4 if "prefix4" in policy else 0)
            assert result["rebase"] == (policy == "standard-rebased-prefix4")
            with np.load(folder / (policy + "-diagnostics.npz")) as diagnostic:
                np.testing.assert_array_equal(result["per_token_ce"], diagnostic["per_token_ce"])
                if input_ids is None:
                    input_ids = diagnostic["input_token_ids"].copy()
                np.testing.assert_array_equal(input_ids, diagnostic["input_token_ids"])
            rows.append({"article_index": index, "title": title, "policy": policy,
                         **token_scores(result), "virtual_before": float(np.mean(result["before_mean_virtual_by_layer_head"])),
                         "virtual_after": float(np.mean(result["after_mean_virtual_by_layer_head"]))})
    write(root / "article_results.json", rows)
    if missing:
        write(root / "INCOMPLETE.json", {"missing": missing, "scope": "No final conclusion until all 12 articles and 5 policies are complete"})
        raise ValueError("incomplete article results")
    values = lambda policy, region: [row[region] for row in rows if row["policy"] == policy]
    metrics = {policy: {region: {"ce": float(np.mean(values(policy, region))),
                    "perplexity": math.exp(float(np.mean(values(policy, region))))} for region in REGIONS} for policy in POLICIES}
    statistics = {"primary_fifo": paired_effect(values("standard-fifo", "post_eviction"), values("query_sink-fifo", "post_eviction"))}
    for policy in ["standard-prefix4", "standard-rebased-prefix4", "query_sink-disabled"]:
        statistics[policy + "_vs_virtual"] = paired_effect(values(policy, "post_eviction"), values("query_sink-fifo", "post_eviction"))
    for region in ["pre_eviction", "beyond_10w"]:
        statistics[region] = paired_effect(values("standard-fifo", region), values("query_sink-fifo", region))
    write(root / "statistics.json", {"metrics": metrics, "paired_effects": statistics, "seed": 1729,
        "scope": "Same 12 articles as the Qwen0.5B confirmation, one larger-model checkpoint pair; article uncertainty does not cover training-seed variability"})
    report = ["# Qwen1.5B streaming replication", "", "Step-975 selected checkpoints, seed 1729. All 12 fixed WikiText articles, 8192 predictions per policy, 512 physical cache entries.", "",
        "| Policy | Post-eviction PPL | Beyond ten windows PPL |", "|---|---:|---:|"]
    for policy in POLICIES:
        report.append(f"| {policy} | {metrics[policy]['post_eviction']['perplexity']:.4f} | {metrics[policy]['beyond_10w']['perplexity']:.4f} |")
    effect = statistics["primary_fifo"]
    report += ["", "Primary paired article-bootstrap result:", "", json.dumps(effect, indent=2), "",
        "Token losses were independently reaggregated. Bootstrap uses 10000 paired article resamples, seed1729. Prefix, rebasing, sink removal and other segments are secondary descriptive comparisons.", "",
        "Only one training pair was completed. This is replication on the existing article set, not new-document confirmation. No cross-seed or cross-corpus conclusion, and no claim of an official StreamingLLM reproduction."]
    (root / "REPORT.md").write_text("\n".join(report))
    write(root / "complete.json", {"complete": True, "articles": 12, "policies": 5, "predictions": 12 * 5 * 8192})
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
