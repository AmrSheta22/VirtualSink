"""Controlled tiny GPU recall pilot, not pretrained-Qwen uptraining.

A single transformer layer makes the FIFO control unable to relay old facts
through cached contextual states: its K/V come directly from token embeddings.
The final query's physical window is independent of the target. A memory bank
can learn to preserve the first token through eviction and random distractors.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch

from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.recurrent_sink import RecurrentSinkLM


def examples(batch, delay, generator, device):
    target = torch.randint(1, 9, (batch,), generator=generator)
    distractors = torch.randint(9, 32, (batch, delay), generator=generator)
    query = torch.zeros(batch, 1, dtype=torch.long)
    tokens = torch.cat((target[:, None], distractors, query), 1)
    return tokens.to(device), target.to(device)


@torch.no_grad()
def measure(model, tokens, target, flags):
    model.eval()
    output = model(tokens, **flags)
    logits = model.base.lm_head(output.hidden[:, -1])
    correct = (logits.argmax(-1) == target).sum().item()
    ce = torch.nn.functional.cross_entropy(logits, target).item()
    return {"examples": len(target), "correct": correct, "accuracy": correct / len(target), "ce": ce}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--memory-bias", type=float, default=-2.)
    parser.add_argument("--input-bias", type=float, default=-4.)
    args = parser.parse_args()
    if args.steps < 1 or not all(torch.isfinite(torch.tensor(x)) for x in (args.memory_bias, args.input_bias)):
        parser.error("invalid diagnostic settings")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("recall pilot requires CUDA inside Slurm")
    root = Path(os.environ["RECURRENT_SINK_ROOT"])
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    config = dict(model_type="qwen2", vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                  hidden_act="silu", rope_theta=10000., rms_norm_eps=1e-6,
                  tie_word_embeddings=True)
    base = NativeLM(config)
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=.02)
    initial = RecurrentSinkLM(base, capacity=8, timescales=(128., 512., 2048.))
    with torch.no_grad():
        for bank in initial.memory:
            bank.memory_bias.fill_(args.memory_bias)
            bank.null_bias.fill_(-6.)
            bank.gate_bias[:, :, bank.width:2 * bank.width].fill_(args.input_bias)
    steps, batch, delay = args.steps, 32, 24
    trained, timings = {}, {}
    for variant in ("memory", "standard"):
        model = copy.deepcopy(initial).cuda()
        groups = model.parameter_groups("full", base_lr=.003, memory_lr=.003)
        flags = {} if variant == "memory" else {"read_memory": False, "read_null": False, "write_memory": False}
        if variant == "standard":
            for parameter in model.memory.parameters():
                parameter.requires_grad_(False)
            groups = groups[:1]
        optimizer = torch.optim.AdamW(groups, foreach=False)
        generator = torch.Generator().manual_seed(1729)
        tick = time.monotonic()
        for step in range(steps):
            tokens, target = examples(batch, delay, generator, "cuda")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, **flags)
            logits = model.base.lm_head(output.hidden[:, -1])
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite recall loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 25 == 0:
                print(json.dumps({"event": "recall_train", "variant": variant,
                                  "step": step + 1, "ce": loss.item(),
                                  "accuracy": (logits.argmax(-1) == target).float().mean().item(),
                                  "elapsed_seconds": time.monotonic() - tick}), flush=True)
            del output, logits, loss
        torch.cuda.synchronize()
        timings[variant] = time.monotonic() - tick
        trained[variant] = model
        del optimizer
    results = {}
    memory = trained["memory"]
    for test_delay in (24, 48, 96):
        generator = torch.Generator().manual_seed(8910 + test_delay)
        tokens, target = examples(512, test_delay, generator, "cuda")
        policies = {"memory": (memory, {}),
                    "zero_memory_values": (memory, {"zero_memory_values": True}),
                    "writes_disabled": (memory, {"write_memory": False}),
                    "reads_disabled": (memory, {"read_memory": False}),
                    "forget_zero": (memory, {"forget_override": 0.}),
                    "trained_standard": (trained["standard"], {"read_memory": False, "read_null": False, "write_memory": False})}
        results[str(test_delay)] = {name: measure(model, tokens, target, flags)
                                   for name, (model, flags) in policies.items()}
    short = results["24"]
    report = {"status": "completed", "kind": "tiny-from-scratch-controlled-recall",
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "seed": 1729, "steps_per_arm": steps, "batch": batch, "capacity": 8,
              "initial_memory_bias": args.memory_bias, "initial_input_bias": args.input_bias,
              "train_delay": delay, "chance_accuracy": .125, "train_seconds": timings,
              "new_parameters": sum(p.numel() for p in memory.memory.parameters()),
              "results": results,
              "recall_gate_supported": short["memory"]["accuracy"] > .5 and
                                       short["memory"]["accuracy"] - short["zero_memory_values"]["accuracy"] > .25,
              "limitations": ["single seed", "one-layer toy architecture", "from scratch; not Qwen uptraining",
                              "different initial virtual mass from the pretrained calibration", "quality is task-specific"]}
    (root / "recall.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "recall_results", **report}), flush=True)


if __name__ == "__main__":
    main()
