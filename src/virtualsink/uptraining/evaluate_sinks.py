"""Slurm-only frozen-checkpoint interventions on identical held-out blocks.

Positive disabled-minus-enabled CE means the trained virtual sinks help.
The paired bootstrap resamples sequence blocks, not independent documents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import TokenBlocks, sha256
from .native import load_native, set_window
from .qwen import disable_sinks
from .train import language_loss


def paired_summary(enabled, disabled):
    difference = np.asarray(disabled) - np.asarray(enabled)
    rng = np.random.default_rng(1729)
    means = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(2000)])
    return {"disabled_minus_enabled_ce": float(difference.mean()),
            "block_bootstrap_95pct_interval": np.quantile(means, [.025, .975]).tolist(),
            "blocks": len(difference), "interpretation": "positive means enabled virtual sinks improve CE; correlated blocks limit interval interpretation"}


@torch.no_grad()
def score(model, data, count, output, label):
    model.eval()
    losses, gates = [], []
    for index in range(min(count, len(data))):
        loss, physical = language_loss(model, data[index][None].cuda(), 64)
        losses.append(float(loss))
        gates.append(torch.stack([g.mean((0, 2)) for g in physical]).cpu().numpy())
        if (index + 1) % 32 == 0:
            print(f"{label}: {index+1}/{min(count,len(data))} blocks, CE={np.mean(losses):.6f}", flush=True)
    if not np.isfinite(losses).all():
        raise ValueError("nonfinite evaluation")
    result = {"ce": float(np.mean(losses)), "perplexity": float(np.exp(np.mean(losses))),
              "per_block_ce": losses, "physical_gate_by_layer_head": np.mean(gates, axis=0).tolist()}
    (output / f"{label}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--standard-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=256)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("a working Slurm GPU allocation is required")
    if args.blocks < 2:
        parser.error("blocks must be >=2")
    for directory in (args.run_dir, args.standard_run):
        if directory is not None and not (directory / "summary.json").exists():
            parser.error(f"training has not completed final evaluation: {directory}")
    saved = torch.load(args.run_dir / "best.pt", map_location="cpu")
    config = saved["config"]
    if config["variant"] != "query_sink":
        parser.error("run-dir must contain the trained query-sink arm")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = config["manifest"]
    model = load_native(manifest["model_path"], config["chunk_size"])
    set_window(model, config.get("window_size", 0) or None)
    path = Path(config["data_dir"]) / manifest["splits"]["test"]["file"]
    if sha256(path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("test checksum mismatch")
    data = TokenBlocks(path, config["seq_len"], model.config.vocab_size)
    metadata = {"checkpoint_sha256": sha256(args.run_dir / "best.pt"), "best_step": saved["step"],
        "training_config": config, "blocks": min(args.blocks, len(data)),
        "scope": "same test blocks; post-training interventions; original weights baseline; no optimizer updates",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    # Match architecture/window for the original-weight baseline.
    model.cuda()
    disable_sinks(model)
    original = score(model, data, args.blocks, args.output_dir, "original_standard")
    model.load_state_dict(saved["model"])
    del saved
    disable_sinks(model, False)
    enabled = score(model, data, args.blocks, args.output_dir, "trained_sinks_enabled")
    disable_sinks(model)
    disabled = score(model, data, args.blocks, args.output_dir, "trained_sinks_disabled")
    disable_sinks(model, False)
    # Remove learned query dependence while preserving trained scalar biases.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.sink_w.zero_()
    static = score(model, data, args.blocks, args.output_dir, "trained_bias_only")
    summary = {"enabled_vs_disabled": paired_summary(enabled["per_block_ce"], disabled["per_block_ce"]),
        "query_vs_bias_only": paired_summary(enabled["per_block_ce"], static["per_block_ce"]),
        "metrics": {k:{"ce":v["ce"], "perplexity":v["perplexity"]} for k,v in
                    (("original_standard", original), ("trained_sinks_enabled", enabled),
                     ("trained_sinks_disabled", disabled), ("trained_bias_only", static))}}
    if args.standard_run:
        baseline = torch.load(args.standard_run / "best.pt", map_location="cpu")
        if baseline["config"]["variant"] != "standard":
            raise ValueError("standard-run must contain the standard-attention control")
        for key in ("manifest", "mode", "seq_len", "window_size", "steps", "batch_size", "accumulation", "seed"):
            if config.get(key) != baseline["config"].get(key):
                raise ValueError(f"unmatched trained standard control: {key}")
        model.load_state_dict(baseline["model"])
        disable_sinks(model)
        metadata["standard_checkpoint_sha256"] = sha256(args.standard_run / "best.pt")
        metadata["standard_best_step"] = baseline["step"]
        del baseline
        control = score(model, data, args.blocks, args.output_dir, "trained_standard_control")
        summary["sink_training_vs_standard_training"] = paired_summary(enabled["per_block_ce"], control["per_block_ce"])
        summary["metrics"]["trained_standard_control"] = {"ce":control["ce"], "perplexity":control["perplexity"]}
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
