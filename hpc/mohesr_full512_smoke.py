"""Run under Slurm: verify the original data and benchmark both full-size arms."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from virtualsink.pretraining.recipe import MODEL_CONFIG, initialize, optimizer_groups
from virtualsink.pretraining.train import language_loss
from virtualsink.uptraining.qwen import disable_sinks

root = Path(os.environ["MOHESR_EXPERIMENT"])
assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
data = root / "data"
manifest = json.loads((data / "manifest.json").read_text())
assert hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest() == "0d8cdb9cb8a074864e481bdec43c5e33f04f69bb2ca01fc0f0224f4baa121cdb"
for source in manifest["sources"].values():
    for split in source["splits"].values():
        assert hashlib.sha256((data / split["file"]).read_bytes()).hexdigest() == split["sha256"]
for split, spec in manifest["splits"].items():
    assert hashlib.sha256((data / (split + "-order.npy")).read_bytes()).hexdigest() == spec["order_sha256"]
print("ALL ORIGINAL DATA HASHES VERIFIED", flush=True)
for variant in ("standard", "query_sink"):
    model = initialize(8, 32, dict(MODEL_CONFIG, max_position_embeddings=512),
                       sink_weight_std=.02, sink_bias=4.).cuda()
    disable_sinks(model, variant == "standard")
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(optimizer_groups(model, variant), lr=.003,
                                 betas=(.9, .95), foreach=False)
    tokens = torch.randint(0, 49152, (1, 513), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    loss, gates = language_loss(model, tokens, 32)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    durations = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        measured, measured_gates = language_loss(model, tokens, 32)
        (measured / 2048).backward()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
        del measured, measured_gates
    seconds = sum(durations[1:]) / len(durations[1:])
    result = dict(variant=variant, microbatch_seconds=seconds,
                  estimated_training_hours=seconds * 2048 * 30 / 3600,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    print(json.dumps(result), flush=True)
    (root / ("benchmark-" + variant + ".json")).write_text(json.dumps(result, indent=2))
    del model, optimizer, tokens, loss, gates
    gc.collect()
    torch.cuda.empty_cache()
