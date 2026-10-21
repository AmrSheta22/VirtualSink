"""Short K80 full-Qwen recurrent-memory correctness and optimizer gate."""

import json
import os
import time
from pathlib import Path

import torch

from virtualsink.uptraining.data import TokenBlocks
from virtualsink.uptraining.native import load_native
from virtualsink.uptraining.recurrent_sink import RecurrentSinkLM
from virtualsink.uptraining.train_recurrent import token_losses


def main():
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires Slurm")
    root = Path(os.environ["RECURRENT_SINK_ROOT"])
    manifest = json.loads(Path(os.environ["RECURRENT_SOURCE_MANIFEST"]).read_text())
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    base = load_native(manifest["model_path"])
    model = RecurrentSinkLM(base, capacity=32, timescales=(128., 512., 2048.)).cuda()
    model.parameter_groups("memory")
    data_dir = Path(os.environ["RECURRENT_SOURCE_MANIFEST"]).parent
    data = TokenBlocks(data_dir / manifest["splits"]["train"]["file"], 128, base.config.vocab_size)
    # Packed tokens are used only for arithmetic profiling, never claimed to
    # be a document-preserving training corpus or a memory-quality evaluation.
    tokens = data[0][:82][None].cuda()
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    calibration = model.calibrate([inputs], target_mass=1e-4)
    print(json.dumps({"event": "calibration", **calibration}), flush=True)
    assert max(max(row) for row in calibration["quantile_mass"]) < 2e-4
    model.eval()
    with torch.no_grad():
        full = model(inputs)
        expected = model.base.lm_head(full.hidden)
        state, logits = None, []
        for token in inputs.unbind(1):
            output, state, _ = model.step(token, state)
            logits.append(output)
        actual = torch.stack(logits, 1)
        parity_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=.003, rtol=1e-4)
        cell_error = max((a.cell - b.cell).abs().max().item() for a, b in zip(full.state.layers, state.layers))
        assert cell_error < .001
        for layer in full.state.layers:
            assert layer.key.shape[2] == model.capacity
            assert torch.isfinite(layer.cell).all() and layer.cell.abs().sum() > 0
    print(json.dumps({"event": "parity", "max_logit_error": parity_error,
                      "max_cell_error": cell_error}), flush=True)
    del full, expected, actual, logits, state, output
    model.train()
    model.checkpointing = True
    optimizer = torch.optim.AdamW(model.parameter_groups("memory"), foreach=False)
    steps = []
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        before = [bank.write_weight.detach().clone() for bank in model.memory]
        # Carry a connected state through a segment boundary and eviction.
        first = model(inputs[:, :40])
        second = model(inputs[:, 40:], first.state)
        losses = torch.cat((token_losses(model, first.hidden, targets[:, :40], 16),
                            token_losses(model, second.hidden, targets[:, 40:], 16)))
        loss = losses.mean()
        assert torch.isfinite(loss)
        loss.backward()
        zero_parameters = []
        for name, parameter in model.memory.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if parameter.grad.count_nonzero() == 0:
                zero_parameters.append(name)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        updates = [(bank.write_weight.detach() - prior).abs().sum().item()
                   for bank, prior in zip(model.memory, before)]
        assert sum(updates) > 0, "memory write matrices must actually update"
        row = {"optimizer_step": iteration + 1, "ce": loss.item(),
               "seconds": time.monotonic() - tick, "gradient_norm": norm.item(),
               "write_update_l1_by_layer": updates, "zero_gradient_parameters": zero_parameters}
        steps.append(row)
        print(json.dumps({"event": "optimizer_step", **row}), flush=True)
        del first, second, losses, loss, before
    report = {"status": "passed", "slurm_job": os.environ["SLURM_JOB_ID"],
              "model": manifest["model"], "revision": manifest["revision"],
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "capacity": model.capacity, "tokens_per_step": inputs.numel(),
              "new_parameters": sum(p.numel() for p in model.memory.parameters()),
              "calibration": calibration, "max_logit_error": parity_error,
              "optimizer_steps": steps, "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "limitation": "correctness/profile only; no long-memory quality conclusion"}
    (root / "smoke.json").write_text(json.dumps(report, indent=2))
    print("RECURRENT_MEMORY_GPU_GATE_PASSED", flush=True)


if __name__ == "__main__":
    main()
