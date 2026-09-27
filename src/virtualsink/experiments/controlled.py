"""Run four attention arms across loss weightings and paired seeds.

Defaults: two losses, three seeds, and 3,000 steps per model.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ..analysis.study import LOSS_MODES, plot_and_report, summarize


def source_hashes() -> dict:
    """Hash every shipped Python module, including split training and analysis code."""
    package_root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(package_root.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/attention-control"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1729, 1730, 1731])
    parser.add_argument("--data-seed", type=int, default=2718)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    root = args.output_dir.resolve()
    if root.exists():
        parser.error("choose a new output directory to preserve existing studies")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "study_config.json").write_text(json.dumps({**vars(args), "output_dir": str(root),
              "architecture": "attention_only", "num_layers": 1, "patience": 0,
              "source_sha256": source_hashes()}, indent=2), encoding="utf-8")
    entries = []
    for mode in LOSS_MODES:
        for seed in args.seeds:
            destination = root / mode / f"seed-{seed}"
            command = [sys.executable, "-u", "-m", "virtualsink.experiments.ablation",
                       "--output-dir", str(destination), "--architecture", "attention_only",
                       "--num-layers", "1", "--loss-mode", mode, "--seed", str(seed),
                       "--patience", "0", "--log-every", "1000"]
            for key in ("device", "data_seed", "steps", "batch_size", "d_model", "seq_len",
                        "n_heads", "validation_samples", "test_samples", "eval_every", "threads"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            log_path = root / "logs" / f"{mode}-{seed}.log"
            print(f"START {mode}, seed={seed}, four models; log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Run failed (exit={completed.returncode}); inspect {log_path}")
            run_summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            entries.append({"loss_mode": mode, "seed": seed, "run_dir": str(destination), "summary": run_summary})
            summary = summarize(root, entries)
            print(f"DONE {mode}, seed={seed}: " + ", ".join(
                f"{v['attn_type']} Gaussian trigger MSE={v['test']['gaussian_trigger_mse']:.6g}"
                for v in run_summary["variants"]), flush=True)
    plot_and_report(root, summary)
    print(f"COMPLETE: {root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
