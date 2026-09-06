"""Plot learned mechanisms without asserting that the desired hypothesis is true."""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from run_ablation import load_checkpoint
from synthetic_dataset import TriggerConditionalDataset
from toy_models import ATTENTION_TYPES, ToyTransformer


LABELS = {"standard": "Standard softmax", "quiet": "Quiet softmax (+1)",
          "static_sink": "Static learned sink", "query_virtual_sink": "Query-conditioned sink"}
COLORS = {"standard": "#6C757D", "quiet": "#CC7A00", "static_sink": "#8064A2",
          "query_virtual_sink": "#007D8A"}


@torch.no_grad()
def mechanism_statistics(model: ToyTransformer, loader: DataLoader) -> Dict[str, dict]:
    """Report every layer/head over held-out examples, with BOS excluded from
    non-trigger mechanism means (BOS itself has no physical alternative).
    MSE in the training runner includes BOS in non-trigger positions.
    """
    totals, counts = {}, {}
    model.eval()
    for batch in loader:
        _, diagnostics = model(batch["x"], return_attention=True)
        trigger = batch["trigger_mask"]
        non_trigger = ~trigger.clone()
        non_trigger[:, 0] = False
        batch_size, length = trigger.shape
        indices = batch["trigger_index"]
        for layer, info in enumerate(diagnostics):
            weights = info["physical_weights"]
            for head in range(weights.shape[1]):
                name = f"layer_{layer}/head_{head}"
                w = weights[:, head]
                trigger_weights = w[torch.arange(batch_size), indices]
                keys = torch.arange(length)[None, :]
                history = (keys >= 1) & (keys < indices[:, None])
                target = history.float() / (indices[:, None] - 1)
                quantities = {
                    "bos_mass_non_trigger": w[..., 0][non_trigger],
                    "physical_mass_non_trigger": w.sum(-1)[non_trigger],
                    "sink_mass_non_trigger": info["sink_probability"][:, head][non_trigger],
                    "sink_mass_trigger": info["sink_probability"][:, head][trigger],
                    "history_mass_trigger": (trigger_weights * history).sum(-1),
                    "history_l1_error_trigger": (trigger_weights - target).abs().sum(-1),
                }
                if "sink_logits" in info:
                    logits = info["sink_logits"][:, head]
                    margin = logits - info["physical_log_partition"][:, head]
                    quantities.update({"sink_logit_non_trigger": logits[non_trigger],
                                       "sink_logit_trigger": logits[trigger],
                                       "sink_margin_non_trigger": margin[non_trigger],
                                       "sink_margin_trigger": margin[trigger],
                                       "positive_sink_fraction_non_trigger": (logits[non_trigger] > 0).float(),
                                       "negative_sink_fraction_trigger": (logits[trigger] < -3).float()})
                totals.setdefault(name, {})
                counts.setdefault(name, {})
                for key, values in quantities.items():
                    assert torch.isfinite(values).all(), f"Non-finite mechanism statistic: {key}"
                    totals[name][key] = totals[name].get(key, 0.0) + values.double().sum().item()
                    counts[name][key] = counts[name].get(key, 0) + values.numel()
    return {name: {key: value / counts[name][key] for key, value in stats.items()}
            for name, stats in totals.items()}


def generate_plots(run_dir: Path, layer: int = 0, head: int = 0, sample_index: int = 0) -> Path:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config.get("threads", 4))
    data = TriggerConditionalDataset(config["test_samples"], config["d_model"],
                                     config["seq_len"], config["seed"] + 303)
    sample = data[sample_index]
    trigger_index = int(sample["trigger_index"])
    length = config["seq_len"]
    examples, all_stats = {}, {}
    for attn_type in ATTENTION_TYPES:
        checkpoint = load_checkpoint(run_dir / attn_type / "best.pt")
        model = ToyTransformer(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        if not 0 <= layer < len(model.blocks) or not 0 <= head < config["n_heads"]:
            raise ValueError("layer/head index is out of range")
        with torch.no_grad():
            _, diagnostics = model(sample["x"].unsqueeze(0), return_attention=True)
        examples[attn_type] = {key: value[0, head].numpy() for key, value in diagnostics[layer].items()}
        all_stats[attn_type] = mechanism_statistics(model, DataLoader(data, batch_size=config["batch_size"]))

    selected_key = f"layer_{layer}/head_{head}"
    standard = all_stats["standard"][selected_key]
    query = all_stats["query_virtual_sink"][selected_key]
    checks = {
        "standard_non_trigger_bos_mass_gt_0.9": standard["bos_mass_non_trigger"] > 0.9,
        "query_non_trigger_bos_mass_lt_0.05": query["bos_mass_non_trigger"] < 0.05,
        "query_non_trigger_physical_mass_lt_0.05": query["physical_mass_non_trigger"] < 0.05,
        "query_trigger_history_mass_gt_0.9": query["history_mass_trigger"] > 0.9,
        "query_non_trigger_mean_logit_gt_0": query["sink_logit_non_trigger"] > 0,
        "query_trigger_mean_logit_lt_minus_3": query["sink_logit_trigger"] < -3,
    }
    report = {"scope": "Held-out test set, all layers/heads; non-trigger mechanism means exclude BOS.",
              "selected_layer": layer, "selected_head": head, "plotted_sample_index": sample_index,
              "plotted_trigger_index": trigger_index, "statistics": all_stats,
              "hypothesis_checks_selected_head": checks,
              "interpretation": "Descriptive checks, not training constraints or a theoretical proof. "
                                "Raw sink logits must be interpreted relative to physical log-partition."}
    (run_dir / "mechanism_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                   encoding="utf-8")
    np.savez_compressed(run_dir / "mechanism_example.npz", trigger_index=trigger_index,
                        **{f"{name}_{key}": value for name, example in examples.items()
                           for key, value in example.items()})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    heat_axes = []
    for column, attn_type in enumerate(ATTENTION_TYPES):
        ax = fig.add_subplot(grid[0, column])
        heat_axes.append(ax)
        im = ax.imshow(examples[attn_type]["physical_weights"], vmin=0, vmax=1,
                       cmap="viridis", origin="upper", interpolation="nearest", aspect="equal")
        ax.axhline(trigger_index, color="#E45756", linestyle="--", linewidth=1)
        ax.set_title(LABELS[attn_type], fontweight="bold")
        ax.set_xlabel("Physical key position")
        if column == 0:
            ax.set_ylabel("Query position")
        ax.set_xticks([0, length // 2, length - 1])
        ax.set_yticks([0, trigger_index, length - 1])
    fig.colorbar(im, ax=heat_axes, label="Physical attention probability", shrink=0.8, pad=0.015)
    ax = fig.add_subplot(grid[1, :])
    example = examples["query_virtual_sink"]
    positions = np.arange(length)
    ax.plot(positions, example["sink_logits"], color=COLORS["query_virtual_sink"],
            marker="o", markersize=3, linewidth=2, label="Query-conditioned sink logit s(Q)")
    ax.plot(positions, example["physical_log_partition"], color="#6C757D", linewidth=1.4,
            linestyle=":", label="Physical log-partition (log-sum-exp)")
    ax.axhline(0, color="#A0A0A0", linewidth=0.7)
    ax.axvline(trigger_index, color="#E45756", linestyle="--", linewidth=1.5,
               label=f"Trigger j = {trigger_index}")
    ax.set_xlabel("Query position")
    ax.set_ylabel("Logit / log-partition")
    ax.set_title("Query-conditioned sink trajectory (observed)", loc="left", fontweight="bold")
    ax.set_xlim(0, length - 1)
    ax.legend(loc="best", frameon=False, ncol=3, fontsize=9)
    fig.suptitle(f"Trigger-conditional attention | layer {layer}, head {head} | "
                 f"held-out sample {sample_index}", fontsize=16, fontweight="bold")
    output = run_dir / "trigger_experiment_results.png"
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix("pdf"), facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for attn_type in ATTENTION_TYPES:
        rows = [json.loads(line) for line in (run_dir / attn_type / "metrics.jsonl").read_text().splitlines()]
        for ax, key, title in zip(axes, ("mse", "trigger_mse", "non_trigger_mse"),
                                   ("Overall validation MSE", "Trigger validation MSE", "Non-trigger validation MSE")):
            ax.plot([row["step"] for row in rows], [row["validation"][key] for row in rows],
                    label=LABELS[attn_type], color=COLORS[attn_type], linewidth=1.8)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Training step")
            ax.grid(alpha=0.15)
    axes[0].set_ylabel("Mean squared error")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(run_dir / "convergence.png", dpi=300, facecolor="white")
    fig.savefig(run_dir / "convergence.pdf", facecolor="white")
    plt.close(fig)
    print(f"Saved plots to {run_dir}; hypothesis checks for {selected_key}: {checks}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/trigger"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    generate_plots(args.run_dir, args.layer, args.head, args.sample_index)


if __name__ == "__main__":
    main()
