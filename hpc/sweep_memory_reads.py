"""Frozen-checkpoint counterfactual read-bias sweep; no optimizer or training.

One layer only: its cell/write inputs come from embeddings, so changed attention
read mass cannot change subsequent memory writes. Bias shifts therefore isolate
read normalization/merge on identical stored content. Forced reads are not a
learned-gate result and are not proposed as pretrained initialization.
"""

import json
import os
from pathlib import Path

import torch

from virtualsink.uptraining.memory_warmup import evaluation
from virtualsink.uptraining.recall_experiment import make_model, SEEDS, atomic_json


@torch.no_grad()
def main():
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("read sweep requires Slurm/CUDA")
    root = Path(os.environ["READ_SWEEP_ROOT"])
    source = Path('/cluster/users/alex116u1/data/memory-warmup-20261013-r1')
    report = {"kind": "frozen-checkpoint-forced-read-diagnostic", "source_root": str(source),
              "seeds": list(SEEDS), "bias_shifts": [0., 5., 10., 15., 20.], "results": {}}
    for seed in SEEDS:
        path = source / "runs" / f"aux_then_joint-seed{seed}" / "final.pt"
        saved = torch.load(path, map_location="cpu")
        model = make_model(seed, -4., "cuda")
        model.load_state_dict(saved["model"])
        model.eval()
        original = [bank.memory_bias.detach().clone() for bank in model.memory]
        data = {}
        for shift in report["bias_shifts"]:
            for bank, bias in zip(model.memory, original):
                bank.memory_bias.copy_(bias + shift)
            result = evaluation(model, "cuda")
            data[str(shift)] = result
            print(json.dumps({"event": "read_sweep", "seed": seed, "shift": shift,
                              "joint_accuracy": result["24"]["joint"]["accuracy"],
                              "aux_accuracy": result["24"]["aux_joint"]["accuracy"],
                              "query_memory_mass": result["24"]["joint"]["query_memory_mass"]}), flush=True)
        report["results"][str(seed)] = data
    atomic_json(root / "read_sweep.json", report)
    print("FROZEN_READ_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
