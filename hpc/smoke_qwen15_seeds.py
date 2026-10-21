"""Real K80 optimizer steps plus auxiliary-only gradients before the six arms."""
import json
import os
from pathlib import Path
import time
import torch
from virtualsink.uptraining.native import load_native, set_window
from virtualsink.uptraining.qwen import parameter_groups, disable_sinks, auxiliary_loss
from virtualsink.uptraining.train import language_loss
from virtualsink.uptraining.data import TokenBlocks


def main():
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
    root = Path(os.environ["QWEN_SEEDS_ROOT"])
    manifest = json.loads((root / "data/manifest.json").read_text())
    results = []
    for variant in ["standard", "query_sink"]:
        torch.manual_seed(1729)
        model = load_native(manifest["model_path"], 32)
        set_window(model, 512)
        groups = parameter_groups(model, "attention")
        if variant == "standard":
            disable_sinks(model)
            for p in groups[1]["params"]:
                p.requires_grad_(False)
            groups = groups[:1]
        model.cuda().train()
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(groups, foreach=False)
        data = TokenBlocks(root / "data/train.bin", 1024, model.config.vocab_size)
        tokens = data[0][None].cuda()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if variant == "query_sink":
            penalty, _ = auxiliary_loss(model.model(tokens).attentions)
            penalty.backward()
            for layer in model.model.layers:
                for p in [layer.self_attn.sink_w, layer.self_attn.sink_b, layer.self_attn.q_proj.weight]:
                    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            optimizer.zero_grad(set_to_none=True)
            del penalty, _, layer, p
        losses = []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(8):
                tokens = data[micro][None].cuda()
                loss, gates = language_loss(model, tokens, 32)
                aux, _ = auxiliary_loss(gates, "anti_collapse" if variant == "query_sink" else "none")
                objective = (loss + .01 * aux) / 8
                assert torch.isfinite(objective)
                objective.backward()
                losses.append(loss.item())
                del objective, loss, gates, aux, _
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            print("SMOKE_OPTIMIZER_STEP", variant, iteration + 1, flush=True)
        row = {"variant": variant, "optimizer_steps": 2, "accumulation": 8,
               "mean_ce": sum(losses) / len(losses), "seconds": time.monotonic() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "auxiliary_gradient_checked": variant == "query_sink"}
        results.append(row)
        print(json.dumps(row), flush=True)
        del optimizer, groups, model, data, tokens
        torch.cuda.empty_cache()
    (root / "smoke.json").write_text(json.dumps(results, indent=2))
    print("FULL_MODEL_GPU_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
