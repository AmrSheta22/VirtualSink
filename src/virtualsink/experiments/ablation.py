"""Train matched attention ablations and evaluate validation-selected checkpoints.

Run with python -m virtualsink.experiments.ablation --help.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..attention import ATTENTION_TYPES
from ..data import TriggerConditionalDataset
from ..training.engine import seed_everything, train_variant
from ..training.metrics import MSEAccumulator


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
    parser.add_argument("--architecture", choices=["transformer", "attention_only"], default="transformer")
    parser.add_argument("--loss-mode", choices=["token_mean", "balanced"], default="token_mean")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Fixed validation/test seed base; defaults to --seed. Training uses --seed.")
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
    if args.architecture == "attention_only" and args.num_layers != 1:
        parser.error("attention_only requires --num-layers 1")
    if args.data_seed is None:
        args.data_seed = args.seed
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
                            args.seq_len, args.data_seed + 202), batch_size=args.batch_size)
    test = DataLoader(TriggerConditionalDataset(args.test_samples, args.d_model,
                      args.seq_len, args.data_seed + 303), batch_size=args.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "torch_version": str(torch.__version__), "resolved_device": str(device)}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run config: {config}", flush=True)
    zero = MSEAccumulator()
    indicator = MSEAccumulator()
    for batch in test:
        zero.update(torch.zeros_like(batch["y"]), batch["y"], batch["trigger_mask"])
        prediction = torch.zeros_like(batch["y"])
        prediction[..., 2] = batch["x"][..., 1]
        indicator.update(prediction, batch["y"], batch["trigger_mask"])
    results = {"zero_predictor_test": zero.compute(), "indicator_predictor_test": indicator.compute(),
               "variants": []}
    for attn_type in ATTENTION_TYPES:
        result = train_variant(attn_type, args, device, validation, test)
        results["variants"].append(result)
        (output_dir / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False),
                                                 encoding="utf-8")
    print("\nBest validation checkpoints evaluated on held-out TEST sequences", flush=True)
    print(f"{'Variant':24s} {'Best step':>9s} {'MSE':>12s} {'Trigger MSE':>14s} {'Non-trigger MSE':>16s}")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attn_type", "best_step", *zero.compute().keys()])
        writer.writeheader()
        for row in results["variants"]:
            m = row["test"]
            print(f"{row['attn_type']:24s} {row['best_step']:9d} {m['mse']:12.6g} "
                  f"{m['trigger_mse']:14.6g} {m['non_trigger_mse']:16.6g}", flush=True)
            writer.writerow({"attn_type": row["attn_type"], "best_step": row["best_step"], **m})
    print(f"Zero predictor test metrics: {results['zero_predictor_test']}", flush=True)
    print("\nGaussian-feature TEST MSE (coordinates 3 onward)", flush=True)
    for row in results["variants"]:
        m = row["test"]
        print(f"{row['attn_type']:24s} trigger={m['gaussian_trigger_mse']:.7g} "
              f"non_trigger={m['gaussian_non_trigger_mse']:.7g} "
              f"balanced={m['gaussian_balanced_mse']:.7g}", flush=True)
    if not args.skip_plots:
        from ..analysis.mechanisms import generate_plots
        generate_plots(output_dir)


if __name__ == "__main__":
    main()
