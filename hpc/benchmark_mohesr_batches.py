"""Slurm-only microbatch benchmark, keeping the global batch at 2048."""
import gc
import json
import os
from pathlib import Path
import time
import torch
from virtualsink.pretraining.recipe import initialize, MODEL_CONFIG, optimizer_groups
from virtualsink.pretraining.train import language_loss
from virtualsink.uptraining.qwen import disable_sinks

assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
root = Path(os.environ["MOHESR_EXPERIMENT"])
rows = []
for variant in ("standard", "query_sink"):
    for batch in (8, 16):
        model = initialize(8, 32, dict(MODEL_CONFIG, max_position_embeddings=512),
                           sink_weight_std=.02, sink_bias=4.).cuda()
        disable_sinks(model, variant == "standard")
        model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(optimizer_groups(model, variant), lr=.003,
                               betas=(.9, .95), foreach=False)
        tokens = torch.randint(0, 49152, (batch, 513), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        durations = []
        try:
            for iteration in range(4):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.monotonic()
                loss, gates = language_loss(model, tokens, 32)
                (loss / (2048 // batch)).backward()
                if iteration == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - start)
                del loss, gates
            seconds = sum(durations[1:]) / 3
            row = dict(variant=variant, microbatch=batch, seconds=seconds,
                       estimated_training_hours=seconds * (2048 // batch) * 30 / 3600,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())
        except torch.cuda.OutOfMemoryError:
            row = dict(variant=variant, microbatch=batch, error="out_of_memory")
        except RuntimeError as error:
            # This cluster's MIG allocator can mask allocation failures with
            # this specific NVML assertion. Other runtime failures must stop.
            if "NVML_SUCCESS == r INTERNAL ASSERT FAILED" not in str(error):
                raise
            row = dict(variant=variant, microbatch=batch, error="MIG allocator NVML assertion")
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, opt, tokens
        gc.collect()
        torch.cuda.empty_cache()
        (root / "microbatch-benchmark.json").write_text(json.dumps(rows, indent=2))
successful = [b for b in (8, 16) if all(any(r.get("variant") == v and r["microbatch"] == b
            and "error" not in r for r in rows) for v in ("standard", "query_sink"))]
assert successful, "No tested batch fits both arms"
best = min(successful, key=lambda b: sum(r.get("estimated_training_hours", 0) for r in rows if r["microbatch"] == b))
(root / "selected-microbatch.txt").write_text(str(best))
print("SELECTED_MICROBATCH", best, flush=True)
