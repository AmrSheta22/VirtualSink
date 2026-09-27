"""Compare matched standard baseline vs. query_sink pretraining runs.

Verifies identical initialization hash, evaluates held-out test block CE with
paired bootstrap, measures physical attention gate emergence, and compares
training/validation trajectories.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


def paired_difference_bootstrap(standard_blocks, query_sink_blocks, seed=1729, n_resamples=2000):
    """Paired block bootstrap difference: standard - query_sink (positive = query_sink is better)."""
    standard_blocks = np.asarray(standard_blocks, dtype=float)
    query_sink_blocks = np.asarray(query_sink_blocks, dtype=float)
    if (standard_blocks.ndim != 1 or standard_blocks.size == 0
            or standard_blocks.shape != query_sink_blocks.shape
            or not np.isfinite(standard_blocks).all()
            or not np.isfinite(query_sink_blocks).all()):
        raise ValueError("Paired test losses must be finite, nonempty, equal-length vectors")
    diff = standard_blocks - query_sink_blocks
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_resamples)])
    return {
        "mean_diff": float(diff.mean()),
        "bootstrap_95pct_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "std_err": float(means.std()),
        "blocks": len(diff),
        "interpretation": "positive mean_diff indicates query_sink achieved lower cross-entropy",
    }


def parse_metrics_log(metrics_path):
    eval_steps = []
    step_metrics = []
    if not Path(metrics_path).exists():
        return eval_steps, step_metrics
    with open(metrics_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "validation" in row and "step" in row:
                eval_steps.append({
                    "step": row["step"],
                    "ce": row["validation"].get("ce"),
                    "perplexity": row["validation"].get("perplexity"),
                })
            elif "lm_ce" in row and "step" in row:
                step_metrics.append({
                    "step": row["step"],
                    "lm_ce": row.get("lm_ce"),
                    "grad_norm": row.get("grad_norm"),
                    "sink_grad_norm": row.get("sink_grad_norm", 0.0),
                    "lr": row.get("lr"),
                })
    return eval_steps, step_metrics


def format_markdown_report(report):
    lines = []
    lines.append("# SmolLM Pretraining: Paired Comparison Report")
    lines.append("")
    lines.append(f"- **Shared Initial Weight SHA256:** `{report['verified_initial_hash']}`")
    lines.append(f"- **Completed Steps:** `{report['completed_steps']}`")
    lines.append("")

    tc = report["test_comparison"]
    pb = tc["paired_bootstrap"]
    ci = pb["bootstrap_95pct_interval"]

    lines.append("## 1. Held-Out Test Evaluation")
    lines.append("")
    lines.append("| Metric | Standard Baseline | Query Sink | Difference (Standard - Query Sink) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| **Cross-Entropy (CE)** | {tc['standard_ce']:.4f} | {tc['query_sink_ce']:.4f} | **{tc['ce_reduction']:+.4f}** |")
    lines.append(f"| **Perplexity (PPL)** | {tc['standard_perplexity']:.2f} | {tc['query_sink_perplexity']:.2f} | {tc['standard_perplexity'] - tc['query_sink_perplexity']:+.2f} |")
    lines.append("")
    lines.append(f"> **Paired Bootstrap (95% CI):** `{pb['mean_diff']:+.4f}` [{ci[0]:+.4f}, {ci[1]:+.4f}] across {pb['blocks']} test blocks.")
    lines.append("")

    dis = report.get("disabled_intervention", {})
    if dis:
        lines.append("## 2. Query Sink Post-Training Intervention (Sinks Disabled)")
        lines.append("")
        lines.append("| Query Sink State | Test CE | Test Perplexity |")
        lines.append("|---|---:|---:|")
        lines.append(f"| **Trained Sinks Enabled** | {dis['enabled_ce']:.4f} | {dis['enabled_ppl']:.2f} |")
        lines.append(f"| **Trained Sinks Disabled** | {dis['disabled_ce']:.4f} | {dis['disabled_ppl']:.2f} |")
        lines.append(f"| **Delta (Disabled - Enabled)** | **{dis['disabled_minus_enabled_ce']:+.4f}** | - |")
        lines.append("")

    gs = report.get("gate_statistics", {})
    ats = report.get("attention_statistics", {})
    if gs or ats:
        lines.append("## 3. Physical Attention Sinks & Attention Gate Emergence")
        lines.append("")
        lines.append("| Metric | Standard Baseline | Query Sink | Note |")
        lines.append("|---|---:|---:|---|")
        if gs:
            lines.append(f"| **Mean Physical Gate ($G$)** | {gs.get('standard_mean_physical_gate', 0):.4f} | {gs.get('query_sink_mean_physical_gate', 0):.4f} | Fraction of attention mass to tokens |")
            lines.append(f"| **Min Physical Gate** | {gs.get('standard_min_physical_gate', 0):.4f} | {gs.get('query_sink_min_physical_gate', 0):.4f} | Lowest single head physical mass |")
            lines.append(f"| **Heads with Gate < 90%** | {gs.get('standard_heads_under_90pct', 0)} | {gs.get('query_sink_heads_under_90pct', 0)} | Heads diverting mass away from tokens |")
            lines.append(f"| **Heads with Gate < 50%** | {gs.get('standard_heads_under_50pct', 0)} | {gs.get('query_sink_heads_under_50pct', 0)} | Heavily gated / sink-dominated heads |")
        if ats:
            if "standard_oldest4_visible_mass" in ats:
                lines.append(f"| **Oldest-4 Visible Tokens Mass** | {ats['standard_oldest4_visible_mass']:.4f} | {ats['query_sink_oldest4_visible_mass']:.4f} | Oldest tokens inside each query's window |")
            lines.append(f"| **First-Token Mass (Late Queries)** | {ats.get('standard_first_token_prob_mean', 0):.4f} | {ats.get('query_sink_first_token_prob_mean', 0):.4f} | Physical sink emergence at pos 0 |")
            lines.append(f"| **First-4 Tokens Mass** | {ats.get('standard_first_four_prob_mean', 0):.4f} | {ats.get('query_sink_first_four_prob_mean', 0):.4f} | Physical sink emergence at pos 0..3 |")
            lines.append(f"| **Virtual Sink Mass Mean** | N/A | {ats.get('query_sink_virtual_prob_mean', 0):.4f} | Virtual sink mass allocated |")
            lines.append(f"| **Virtual Sink Mass Max** | N/A | {ats.get('query_sink_virtual_prob_max', 0):.4f} | Peak virtual sink head usage |")
        lines.append("")

    traj = report.get("validation_trajectory", [])
    if traj:
        lines.append("## 4. Validation Loss Trajectory Across Steps")
        lines.append("")
        lines.append("| Step | Standard CE | Query Sink CE | Delta (Std - Query) | Standard PPL | Query Sink PPL |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for t in traj:
            diff_str = f"{t['ce_diff']:+.4f}" if t.get("ce_diff") is not None else "N/A"
            lines.append(f"| {t['step']} | {t['standard_ce']:.4f} | {t['query_sink_ce']:.4f} | {diff_str} | {t['standard_ppl']:.2f} | {t['query_sink_ppl']:.2f} |")
        lines.append("")

    return "\n".join(lines)


def compare_runs(standard_dir, query_sink_dir, output_dir=None):
    standard_dir = Path(standard_dir)
    query_sink_dir = Path(query_sink_dir)

    std_status_file = standard_dir / "status.json"
    sink_status_file = query_sink_dir / "status.json"
    if not std_status_file.exists() or not sink_status_file.exists():
        raise FileNotFoundError(f"Missing status.json: {standard_dir} or {query_sink_dir}")

    std_status = json.loads(std_status_file.read_text())
    sink_status = json.loads(sink_status_file.read_text())
    if std_status.get("status") != "complete" or sink_status.get("status") != "complete":
        raise RuntimeError(f"Runs incomplete: standard={std_status.get('status')}, query_sink={sink_status.get('status')}")

    std_config = json.loads((standard_dir / "config.json").read_text())
    sink_config = json.loads((query_sink_dir / "config.json").read_text())

    std_hash = std_config.get("shared_initial_weights_sha256")
    sink_hash = sink_config.get("shared_initial_weights_sha256")
    if not std_hash or not sink_hash or std_hash != sink_hash:
        raise ValueError(f"Initial shared weight SHA256 mismatch! standard={std_hash}, query_sink={sink_hash}")

    required = ("manifest_sha256", "steps", "model_seed", "data_seed", "initialization",
                "global_batch", "sequence_length", "auxiliary", "gamma", "precision",
                "lr", "betas", "epsilon", "weight_decay", "clip_grad", "torch",
                "source_sha256", "recipe", "microbatch", "chunk_size", "eval_blocks", "eval_every")
    for key in required:
        if key not in std_config or key not in sink_config or std_config[key] is None or sink_config[key] is None:
            raise ValueError(f"Missing required comparison setting: {key}")
    if not std_config["manifest_sha256"] or not std_config["source_sha256"]:
        raise ValueError("Missing data or source provenance")
    if std_config.get("variant") != "standard" or sink_config.get("variant") != "query_sink":
        raise ValueError("Expected standard and query_sink variants")
    # Paths and checkpoint timing do not change the paired training recipe.
    excluded = {"variant", "output_dir", "data_dir", "save_seconds", "max_hours"}
    differences = [key for key in set(std_config) | set(sink_config)
                   if key not in excluded and std_config.get(key) != sink_config.get(key)]
    if differences:
        raise ValueError(f"Paired configuration mismatch: {sorted(differences)}")

    std_summary = json.loads((standard_dir / "summary.json").read_text())
    sink_summary = json.loads((query_sink_dir / "summary.json").read_text())
    expected_steps = std_config["steps"]
    if type(expected_steps) is not int or expected_steps < 2:
        raise ValueError("Invalid configured step budget")
    for record in (std_status, sink_status, std_summary, sink_summary):
        if type(record.get("step")) is not int or record["step"] != expected_steps:
            raise ValueError("Completed steps must match both configured budgets and summaries")

    std_test = std_summary["final_test"]
    sink_test = sink_summary["final_test"]

    test_comparison = paired_difference_bootstrap(
        std_test["per_block_ce"], sink_test["per_block_ce"]
    )

    disabled_intervention = {}
    if "final_test_sinks_disabled" in sink_summary:
        dis_test = sink_summary["final_test_sinks_disabled"]
        disabled_intervention = {
            "enabled_ce": sink_test["ce"],
            "disabled_ce": dis_test["ce"],
            "disabled_minus_enabled_ce": sink_summary.get("disabled_minus_enabled_ce", dis_test["ce"] - sink_test["ce"]),
            "enabled_ppl": sink_test["perplexity"],
            "disabled_ppl": dis_test["perplexity"],
            "interpretation": "positive disabled_minus_enabled_ce confirms trained virtual sinks are actively used and beneficial at test time"
        }

    std_gates = np.array(std_test.get("physical_gate_by_layer_head", []))
    sink_gates = np.array(sink_test.get("physical_gate_by_layer_head", []))
    gate_stats = {}
    if std_gates.size > 0 and sink_gates.size > 0:
        gate_stats = {
            "standard_mean_physical_gate": float(std_gates.mean()),
            "standard_min_physical_gate": float(std_gates.min()),
            "standard_heads_under_90pct": int((std_gates < 0.90).sum()),
            "standard_heads_under_50pct": int((std_gates < 0.50).sum()),
            "query_sink_mean_physical_gate": float(sink_gates.mean()),
            "query_sink_min_physical_gate": float(sink_gates.min()),
            "query_sink_heads_under_90pct": int((sink_gates < 0.90).sum()),
            "query_sink_heads_under_50pct": int((sink_gates < 0.50).sum()),
        }

    std_attn = std_summary.get("attention", {})
    sink_attn = sink_summary.get("attention", {})
    attn_stats = {}
    if "by_layer_head" in std_attn and "by_layer_head" in sink_attn:
        std_by_lh = np.array(std_attn["by_layer_head"])
        sink_by_lh = np.array(sink_attn["by_layer_head"])
        attn_stats = {
            "standard_first_token_prob_mean": float(std_by_lh[:, :, 0].mean()),
            "standard_first_four_prob_mean": float(std_by_lh[:, :, 1].mean()),
            "query_sink_first_token_prob_mean": float(sink_by_lh[:, :, 0].mean()),
            "query_sink_first_four_prob_mean": float(sink_by_lh[:, :, 1].mean()),
            "query_sink_virtual_prob_mean": float(sink_by_lh[:, :, 2].mean()),
            "query_sink_virtual_prob_max": float(sink_by_lh[:, :, 2].max()),
        }
        if std_by_lh.shape[-1] >= 7 and sink_by_lh.shape[-1] >= 7:
            attn_stats.update(standard_oldest4_visible_mass=float(std_by_lh[:, :, 6].mean()),
                              query_sink_oldest4_visible_mass=float(sink_by_lh[:, :, 6].mean()))

    std_evals, _ = parse_metrics_log(standard_dir / "metrics.jsonl")
    sink_evals, _ = parse_metrics_log(query_sink_dir / "metrics.jsonl")
    val_trajectory = []
    eval_dict_sink = {e["step"]: e for e in sink_evals}
    for e in std_evals:
        s = e["step"]
        if s in eval_dict_sink:
            val_trajectory.append({
                "step": s,
                "standard_ce": e["ce"],
                "standard_ppl": e["perplexity"],
                "query_sink_ce": eval_dict_sink[s]["ce"],
                "query_sink_ppl": eval_dict_sink[s]["perplexity"],
                "ce_diff": (e["ce"] - eval_dict_sink[s]["ce"]) if (e["ce"] is not None and eval_dict_sink[s]["ce"] is not None) else None
            })

    report = {
        "verified_initial_hash": std_hash,
        "completed_steps": std_status.get("step"),
        "test_comparison": {
            "standard_ce": std_test["ce"],
            "standard_perplexity": std_test["perplexity"],
            "query_sink_ce": sink_test["ce"],
            "query_sink_perplexity": sink_test["perplexity"],
            "ce_reduction": std_test["ce"] - sink_test["ce"],
            "paired_bootstrap": test_comparison,
        },
        "disabled_intervention": disabled_intervention,
        "gate_statistics": gate_stats,
        "attention_statistics": attn_stats,
        "validation_trajectory": val_trajectory,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(report, indent=2))
        md_content = format_markdown_report(report)
        (out / "comparison.md").write_text(md_content)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Experiment root containing runs/standard and runs/query_sink")
    parser.add_argument("--standard-dir", type=Path, help="Standard baseline run directory")
    parser.add_argument("--query-sink-dir", type=Path, help="Query sink run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for comparison results")
    args = parser.parse_args()

    if args.root:
        standard_dir = args.root / "runs" / "standard"
        query_sink_dir = args.root / "runs" / "query_sink"
        output_dir = args.output_dir or (args.root / "comparison")
    elif args.standard_dir and args.query_sink_dir:
        standard_dir = args.standard_dir
        query_sink_dir = args.query_sink_dir
        output_dir = args.output_dir or query_sink_dir.parent / "comparison"
    else:
        parser.error("Specify either --root or both --standard-dir and --query-sink-dir")

    report = compare_runs(standard_dir, query_sink_dir, output_dir)
    md = format_markdown_report(report)
    print("\n" + md + "\n", flush=True)
    if output_dir:
        print(f"Results written to {output_dir / 'comparison.json'} and {output_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
