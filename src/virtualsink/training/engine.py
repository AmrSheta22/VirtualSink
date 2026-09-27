"""Evaluation and single-variant optimization, independent of the CLI."""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import TriggerConditionalDataset
from ..models import build_model
from .checkpoints import load_checkpoint
from .metrics import MSEAccumulator, regression_loss


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
    model = build_model(dict(architecture=args.architecture, attn_type=attn_type,
                             d_model=args.d_model, n_heads=args.n_heads,
                             num_layers=args.num_layers)).to(device)
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
            loss = regression_loss(prediction, y, mask, args.loss_mode)
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
            selection_metric = "mse" if args.loss_mode == "token_mean" else "balanced_mse"
            val_loss = validation_metrics[selection_metric]
            if val_loss < best:
                best, best_step = val_loss, step
                # CPU tensors keep checkpoints portable between HPC and local plotting.
                checkpoint = {"model_config": model.config,
                              "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              "step": step, "validation": validation_metrics,
                              "selection_metric": selection_metric,
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
            "selection_metric": checkpoint["selection_metric"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation": checkpoint["validation"], "test": evaluate(model, test, device),
            "elapsed_seconds": time.monotonic() - started}
