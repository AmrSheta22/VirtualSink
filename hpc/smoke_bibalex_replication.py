"""Slurm-only full-model check of the fixed batch on Bibalex's K80 runtime."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from virtualsink.pretraining.recipe import initialize, MODEL_CONFIG, optimizer_groups
from virtualsink.pretraining.train import language_loss
from virtualsink.uptraining.qwen import disable_sinks

assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
root = Path(os.environ["SPLIT_EXPERIMENT"])
data = root / "data"
raw = (data / "manifest.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "0d8cdb9cb8a074864e481bdec43c5e33f04f69bb2ca01fc0f0224f4baa121cdb"
manifest = json.loads(raw)
for source in manifest["sources"].values():
    for spec in source["splits"].values():
        assert hashlib.sha256((data / spec["file"]).read_bytes()).hexdigest() == spec["sha256"]
for split, spec in manifest["splits"].items():
    assert hashlib.sha256((data / (split + "-order.npy")).read_bytes()).hexdigest() == spec["order_sha256"]
variant = sys.argv[1]
model = initialize(8, 32, dict(MODEL_CONFIG, max_position_embeddings=512),
                   sink_weight_std=.02, sink_bias=4.).cuda()
disable_sinks(model, variant == "standard")
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(optimizer_groups(model, variant), lr=.003, betas=(.9, .95), foreach=False)
tokens = torch.randint(0, 49152, (16, 513), device="cuda")
start = time.monotonic()
loss, gates = language_loss(model, tokens, 32)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
opt.step()
torch.cuda.synchronize()
print(json.dumps(dict(event="FULL_MODEL_GPU_CHECK_PASSED", variant=variant,
    data_hashes_verified=True, microbatch=16, ce=loss.item(), seconds=time.monotonic()-start,
    peak_gpu_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(),
    torch=str(torch.__version__))), flush=True)
