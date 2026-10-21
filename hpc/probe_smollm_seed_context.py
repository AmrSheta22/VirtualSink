"""Slurm-only exploratory checkpoint probe; never writes training outputs."""
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

EXPERIMENTS = [
    ("/cluster/users/alex116u1/data/smollm-full512-replication-20261009", s) for s in (8,9,10)
] + [("/cluster/users/alex116u1/data/smollm-warm70-20261014", s) for s in (11,12,13)]


def main():
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
    root_name, seed = EXPERIMENTS[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    root = Path(root_name)
    # Read the exact frozen decoder/data implementation of each experiment.
    sys.path.insert(0, str(root / "release/src"))
    from virtualsink.pretraining.recipe import initialize, MODEL_CONFIG
    from virtualsink.pretraining.data import MixedBlocks
    from virtualsink.pretraining.train import language_loss
    from virtualsink.uptraining.native import set_window
    from virtualsink.uptraining.qwen import disable_sinks
    from virtualsink.uptraining.data import sha256
    output = Path(os.environ["SEED_ANALYSIS_ROOT"])
    test = MixedBlocks(root / "data", "test")
    # Identical evenly spaced held-out blocks for all checkpoints/experiments.
    indices = np.linspace(0, len(test)-1, 32, dtype=int).tolist()
    manifest = json.loads((root / "data/manifest.json").read_text())
    # Training unigram estimates per source, plus a shuffled 16k-token test probe.
    counts = np.zeros(49152, dtype=np.int64)
    for spec in manifest["sources"].values():
        ids = np.memmap(root / "data" / spec["splits"]["train"]["file"], dtype="<u4", mode="r")
        counts += np.bincount(ids, minlength=49152)
    logprob = np.log((counts + 1) / (counts.sum() + len(counts)))
    unigram = [float(-logprob[test[i].numpy()[1:]].mean()) for i in indices]
    result = dict(seed=seed, experiment_root=str(root), blocks=indices, tokens_per_condition=32*512,
                  unigram_ce=float(np.mean(unigram)), arms={},
                  note="Exploratory fixed 32 held-out blocks, all 512 target positions. Self-only probe removes all preceding-token attention at every layer. Sink-disabled comparison controls the gate-normalizer change. Off-distribution intervention, not a causal training experiment.")
    for arm in ("standard", "query_sink"):
        run = root / f"seeds/seed{seed}/runs/{arm}"
        config = json.loads((run / "config.json").read_text())
        model = initialize(seed, 32, dict(MODEL_CONFIG, max_position_embeddings=512),
                           sink_weight_std=.02, sink_bias=4.)
        payload = torch.load(run / "final.pt", map_location="cpu")
        assert payload["config"] == config and payload["progress"]["step"] == config["steps"]
        model.load_state_dict(payload["model"])
        del payload
        model.cuda().eval()
        values = {}
        with torch.no_grad():
            for name, disabled, window in (("normal",arm=="standard",None),
                    ("disabled_full",True,None),("disabled_self_only",True,1)):
                disable_sinks(model, disabled)
                set_window(model, window)
                losses = [language_loss(model, test[i][None].cuda(), 32)[0].item() for i in indices]
                values[name] = dict(ce=float(np.mean(losses)), per_block_ce=losses)
        result["arms"][arm] = values
        print(json.dumps(dict(seed=seed, arm=arm, ce={k:v["ce"] for k,v in values.items()})), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (output / f"context-seed{seed}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
