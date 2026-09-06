"""Train four matched ablations, select validation checkpoints, and plot results.

Example (inside a Slurm allocation on this repository's HPC):
    python -u run_ablation.py --output-dir runs/trigger --device cpu
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from synthetic_dataset import TriggerConditionalDataset
from toy_models import ATTENTION_TYPES, ToyTransformer


class MSEAccumulator:
    """Accumulate element-weighted MSE; never average unequal batch means."""

    def __init__(self) -> None:
        self.sums = {"mse": 0.0, "trigger_mse": 0.0, "non_trigger_mse": 0.0}
        self.counts = dict.fromkeys(self.sums, 0)

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               trigger_mask: torch.Tensor) -> None:
        error = (prediction.detach().float() - target.float()).square().mean(dim=-1)
        assert torch.isfinite(error).all(), "Non-finite prediction or squared error"
        for name, values in (("mse", error), ("trigger_mse", error[trigger_mask]),
                             ("non_trigger_mse", error[~trigger_mask])):
            self.sums[name] += values.double().sum().item()
            self.counts[name] += values.numel()

    def compute(self) -> Dict[str, float]:
        if not all(self.counts.values()):
            raise ValueError("metrics require both trigger and non-trigger positions")
        return {name: self.sums[name] / self.counts[name] for name in self.sums}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    metrics = MSEAccumulator()
    try:
        for batch in loader:
            x, y, mask = [batch[key].to(device) for key in ("x", "y", "trigger_mask")]
            metrics.update(model(x), y, mask)
    finally:
        model.train(was_training)
    return metrics.compute()


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> dict:
    # The compatibility fallback is for this project's own checkpoints on torch 1.12.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_variant(attn_type: str, args: argparse.Namespace, device: torch.device,
                  validation: DataLoader, test: DataLoader) -> dict:
    # All constructors consume identical initialization draws before removing
    # unused sink parameters, so every shared tensor starts identically.
    seed_everything(args.seed)
    model = ToyTransformer(attn_type, args.d_model, args.n_heads,
                           num_layers=args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training = DataLoader(TriggerConditionalDataset(args.steps * args.batch_size,
                          args.d_model, args.seq_len, args.seed + 101),
                          batch_size=args.batch_size, shuffle=False, num_workers=0)
    variant_dir = Path(args.output_dir) / attn_type
    variant_dir.mkdir(parents=True, exist_ok=False)
    best, best_step, stale = math.inf, 0, 0
    plateau_reference = math.inf
    started = time.monotonic()
    interval_metrics = MSEAccumulator()
    with (variant_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        initial = evaluate(model, validation, device)
        log.write(json.dumps({"step": 0, "validation": initial}) + "\n")
        log.flush()
        print(f"{attn_type}: initial validation={initial}", flush=True)
        for step, batch in enumerate(training, start=1):
            model.train()
            x, y, mask = [batch[key].to(device) for key in ("x", "y", "trigger_mask")]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = (prediction - y).square().mean()
            assert torch.isfinite(loss), f"{attn_type} step {step}: non-finite loss"
            loss.backward()
            # This also fails on NaN/Inf gradients, rather than silently clipping them.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm,
                                                error_if_nonfinite=True)
            optimizer.step()
            interval_metrics.update(prediction, y, mask)
            if step % args.log_every == 0 or step == 1:
                print(f"{attn_type} step={step}/{args.steps} loss={loss.item():.7g} "
                      f"grad_norm={grad_norm.item():.4g} elapsed={time.monotonic()-started:.1f}s",
                      flush=True)
            if step % args.eval_every != 0 and step != args.steps:
                continue
            validation_metrics = evaluate(model, validation, device)
            row = {"step": step, "train": interval_metrics.compute(),
                   "validation": validation_metrics, "elapsed_seconds": time.monotonic() - started}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            interval_metrics = MSEAccumulator()
            val_loss = validation_metrics["mse"]
            if val_loss < best:
                best, best_step = val_loss, step
                # CPU tensors keep checkpoints portable between HPC and local plotting.
                checkpoint = {"model_config": model.config,
                              "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "step": step, "validation": validation_metrics,
                              "run_config": vars(args)}
                torch.save(checkpoint, variant_dir / "best.pt")
            if val_loss < plateau_reference - args.min_delta:
                plateau_reference, stale = val_loss, 0
            else:
                stale += 1
            print(f"{attn_type} validation step={step}: {validation_metrics}; best_step={best_step}",
                  flush=True)
            if args.patience and step >= args.min_steps and stale >= args.patience:
                print(f"{attn_type}: validation plateau, stopping at step {step}", flush=True)
                break
    checkpoint = load_checkpoint(variant_dir / "best.pt", device)
    model.load_state_dict(checkpoint["model_state"])
    return {"attn_type": attn_type, "best_step": best_step, "steps_trained": step,
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation": checkpoint["validation"], "test": evaluate(model, test, device),
            "elapsed_seconds": time.monotonic() - started}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/trigger")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10, help="0 disables plateau stopping")
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("steps", "batch_size", "eval_every", "log_every", "validation_samples",
                 "test_samples", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.patience < 0 or args.min_delta < 0 or args.min_steps < 0:
        parser.error("patience, min-delta and min-steps must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        parser.error("lr/max-grad-norm must be positive; weight-decay must be nonnegative")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a fresh run directory")
    if not args.skip_plots:
        import matplotlib  # Fail before spending time training if plotting is unavailable.
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    validation = DataLoader(TriggerConditionalDataset(args.validation_samples, args.d_model,
                            args.seq_len, args.seed + 202), batch_size=args.batch_size)
    test = DataLoader(TriggerConditionalDataset(args.test_samples, args.d_model,
                      args.seq_len, args.seed + 303), batch_size=args.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "torch_version": str(torch.__version__), "resolved_device": str(device)}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run config: {config}", flush=True)
    zero = MSEAccumulator()
    for batch in test:
        zero.update(torch.zeros_like(batch["y"]), batch["y"], batch["trigger_mask"])
    results = {"zero_predictor_test": zero.compute(), "variants": []}
    for attn_type in ATTENTION_TYPES:
        result = train_variant(attn_type, args, device, validation, test)
        results["variants"].append(result)
        (output_dir / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False),
                                                 encoding="utf-8")
    print("\nBest validation checkpoints evaluated on held-out TEST sequences", flush=True)
    print(f"{'Variant':24s} {'Best step':>9s} {'MSE':>12s} {'Trigger MSE':>14s} {'Non-trigger MSE':>16s}")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attn_type", "best_step", "mse",
                                                    "trigger_mse", "non_trigger_mse"])
        writer.writeheader()
        for row in results["variants"]:
            m = row["test"]
            print(f"{row['attn_type']:24s} {row['best_step']:9d} {m['mse']:12.6g} "
                  f"{m['trigger_mse']:14.6g} {m['non_trigger_mse']:16.6g}", flush=True)
            writer.writerow({"attn_type": row["attn_type"], "best_step": row["best_step"], **m})
    print(f"Zero predictor test metrics: {results['zero_predictor_test']}", flush=True)
    if not args.skip_plots:
        from visualize_mechanisms import generate_plots
        generate_plots(output_dir)


if __name__ == "__main__":
    main()
