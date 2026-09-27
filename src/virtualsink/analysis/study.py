"""Descriptive statistics and reports for paired controlled studies."""

import csv
import json
import statistics
from pathlib import Path

from ..attention import ATTENTION_TYPES


METRICS = ("mse", "trigger_mse", "non_trigger_mse", "balanced_mse",
           "gaussian_trigger_mse", "gaussian_non_trigger_mse", "gaussian_balanced_mse",
           "indicator_trigger_mse")
LOSS_MODES = ("token_mean", "balanced")


def describe(values: list) -> dict:
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def summarize(root: Path, entries: list) -> dict:
    """Aggregate only completed runs; retain individual seeds and checkpoints."""
    summary = {"runs": entries, "conditions": {}, "paired_query_minus_standard": {}}
    for loss_mode in LOSS_MODES:
        subset = [entry for entry in entries if entry["loss_mode"] == loss_mode]
        if not subset:
            continue
        summary["conditions"][loss_mode] = {}
        for name in ATTENTION_TYPES:
            rows = [next(v for v in entry["summary"]["variants"] if v["attn_type"] == name)
                    for entry in subset]
            group = {key: describe([row["test"][key] for row in rows]) for key in METRICS}
            group["gaussian_trigger_ratio_to_zero"] = describe([
                row["test"]["gaussian_trigger_mse"] /
                entry["summary"]["zero_predictor_test"]["gaussian_trigger_mse"]
                for row, entry in zip(rows, subset)])
            summary["conditions"][loss_mode][name] = group
        summary["paired_query_minus_standard"][loss_mode] = {}
        for metric in METRICS:
            differences = []
            for entry in subset:
                variants = {v["attn_type"]: v["test"] for v in entry["summary"]["variants"]}
                differences.append(variants["query_virtual_sink"][metric] - variants["standard"][metric])
            summary["paired_query_minus_standard"][loss_mode][metric] = describe(differences)
    (root / "study_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def plot_and_report(root: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(ATTENTION_TYPES)
    labels = ["Standard", "Quiet", "Static sink", "Query sink"]
    colors = ["#6C757D", "#CC7A00", "#8064A2", "#007D8A"]
    titles = {"token_mean": "Original token-averaged loss", "balanced": "Balanced trigger / non-trigger loss"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, mode in enumerate(LOSS_MODES):
        group = summary["conditions"][mode]
        for column, metric in enumerate(("gaussian_trigger_mse", "gaussian_non_trigger_mse")):
            ax = axes[row, column]
            for index, name in enumerate(names):
                values = group[name][metric]
                ax.errorbar(index, values["mean"], yerr=values["std"] or 0,
                            color=colors[index], marker="s", markersize=8, capsize=5)
                offsets = np.linspace(-0.09, 0.09, values["n"])
                ax.scatter(index + offsets, values["values"], color=colors[index], s=20, alpha=0.6)
            if column == 0:
                zero = summary["runs"][0]["summary"]["zero_predictor_test"][metric]
                ax.axhline(zero, color="#333333", linestyle="--", linewidth=1, label="Zero / indicator-only predictor")
                ax.legend(frameon=False, fontsize=8)
            ax.set_xticks(range(len(names)), labels)
            ax.set_ylabel("Gaussian-feature MSE (lower is better)")
            ax.set_title(titles[mode] + ("\nTrigger positions" if column == 0 else "\nNon-trigger positions"))
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Attention-only control | mean ± sample SD; dots are training seeds", fontsize=14)
    fig.savefig(root / "controlled_comparison.png", dpi=300, facecolor="white")
    fig.savefig(root / "controlled_comparison.pdf", facecolor="white")
    plt.close(fig)

    fields = ["loss_mode", "attn_type", "metric", "mean", "std", "n"]
    with (root / "controlled_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, groups in summary["conditions"].items():
            for name, metrics in groups.items():
                for key, stats in metrics.items():
                    writer.writerow(dict(loss_mode=mode, attn_type=name, metric=key,
                                         **{k: stats[k] for k in ("mean", "std", "n")}))
    lines = ["# Attention-only control", "",
             "One attention layer; no residual, normalization, MLP, or extra prediction head. "
             "All Q/K/V/output projections remain trainable. Each loss uses all 64 coordinates. "
             "Only the weighting of trigger vs non-trigger tokens changes between losses.", "",
             "Each point is a separately initialized training seed, paired across all attention arms and losses. "
             "Validation and test examples are fixed across seeds. Checkpoints minimize their own validation "
             "training objective. No checkpoints or settings are selected using test metrics.", "",
             "| Loss | Attention | Trigger Gaussian MSE | Non-trigger Gaussian MSE | Balanced Gaussian MSE |",
             "|---|---|---:|---:|---:|"]
    for mode, groups in summary["conditions"].items():
        for name, metrics in groups.items():
            cells = []
            for key in ("gaussian_trigger_mse", "gaussian_non_trigger_mse", "gaussian_balanced_mse"):
                stats = metrics[key]
                cells.append(f"{stats['mean']:.6g} ± {stats['std']:.2g}" if stats["std"] is not None
                             else f"{stats['mean']:.6g} (one seed)")
            lines.append(f"| {mode} | {name} | " + " | ".join(cells) + " |")
    zero = summary["runs"][0]["summary"]["zero_predictor_test"]["gaussian_trigger_mse"]
    lines.extend(["", f"Zero and indicator-only predictors have trigger Gaussian MSE **{zero:.6g}** "
                  "and zero non-trigger Gaussian MSE. Error bars describe seed variability, not confidence intervals.",
                  "", "See per-run mechanism_summary.json for every head's sink/BOS mass, and per-run "
                  "trigger_experiment_results.png for a fixed held-out example. Low regression loss alone "
                  "does not establish that sink offloading caused the improvement. Trainable projections "
                  "can still shrink values; BOS has Gaussian features and is not a guaranteed zero-valued token."])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
