"""Preregistered initialization/budget/curriculum study for recurrent recall.

Short, Slurm-only random one-layer experiments. No pretrained-Qwen gain claim.
All arms share seeds, architecture, example order and held-out test examples.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from .native import NativeLM
from .recurrent_sink import RecurrentSinkLM


SEEDS = (1729, 1730, 1731)
ARMS = (
    dict(name="low_weak_fixed", mass=1e-4, input_bias=-4., curriculum=False),
    dict(name="low_strong_fixed", mass=1e-4, input_bias=-1., curriculum=False),
    dict(name="high_weak_fixed", mass=.5, input_bias=-4., curriculum=False),
    dict(name="high_strong_fixed", mass=.5, input_bias=-1., curriculum=False),
    dict(name="low_weak_curriculum", mass=1e-4, input_bias=-4., curriculum=True),
    dict(name="low_strong_curriculum", mass=1e-4, input_bias=-1., curriculum=True),
    dict(name="standard_fixed", mass=0., input_bias=-4., curriculum=False),
    dict(name="standard_curriculum", mass=0., input_bias=-4., curriculum=True),
)
COMMON = dict(steps=1000, snapshots=[0, 300, 1000], batch=32, capacity=8,
              timescales=[128., 512., 2048.], lr=.003, train_delay=24,
              evaluation_delays=[24, 48, 96], evaluation_examples=512,
              curriculum_steps=300, calibration_examples=64)


def matrix():
    return [dict(index=i * len(ARMS) + j, seed=seed, **arm)
            for i, seed in enumerate(SEEDS) for j, arm in enumerate(ARMS)]


def train_delay(arm, step):
    """Steps are one-based; even the first curriculum query follows eviction."""
    if not 1 <= step <= COMMON["steps"]:
        raise ValueError("step outside frozen budget")
    if not arm["curriculum"]:
        return COMMON["train_delay"]
    progress = min(step - 1, COMMON["curriculum_steps"] - 1)
    return COMMON["capacity"] + (COMMON["train_delay"] - COMMON["capacity"]) * progress // (COMMON["curriculum_steps"] - 1)


def examples(batch, delay, generator, device):
    target = torch.randint(1, 9, (batch,), generator=generator)
    distractors = torch.randint(9, 32, (batch, delay), generator=generator)
    return torch.cat((target[:, None], distractors, torch.zeros(batch, 1, dtype=torch.long)), 1).to(device), target.to(device)


def make_model(seed, input_bias, device):
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    config = dict(model_type="qwen2", vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                  hidden_act="silu", rope_theta=10000., rms_norm_eps=1e-6,
                  tie_word_embeddings=True)
    base = NativeLM(config)
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=.02)
    model = RecurrentSinkLM(base, COMMON["capacity"], COMMON["timescales"]).to(device)
    with torch.no_grad():
        for bank in model.memory:
            bank.gate_bias[:, :, bank.width:2 * bank.width].fill_(input_bias)
    return model


@torch.no_grad()
def initialize_mass(model, seed, target_mass, device):
    """Use the production conservative calibrator, then a toy-only bias shift.

    Since this model has one layer, its QKV/write inputs depend only on token
    embeddings. Thus shifting all virtual biases changes joint mass without
    changing prospective logits or cell state. Never generalize this shortcut
    to a multi-layer pretrained model; its production calibrator remains capped.
    """
    if len(model.memory) != 1 or not 0 <= target_mass < 1:
        raise ValueError("mass initializer requires a one-layer toy model")
    generator = torch.Generator().manual_seed(seed + 50000)
    tokens, _ = examples(COMMON["calibration_examples"], COMMON["train_delay"], generator, device)
    if target_mass:
        report = model.calibrate([tokens], target_mass=1e-4)
        shift = math.log(target_mass / (1 - target_mass)) - math.log(1e-4 / (1 - 1e-4))
        for bank in model.memory:
            bank.null_bias.add_(shift)
            bank.memory_bias.add_(shift)
        result = model(tokens, diagnostics=True)
        control = model(tokens, read_memory=False, read_null=False)
        mass = result.diagnostics["null_mass"] + result.diagnostics["memory_mass"]
        per_head_max = mass.amax(dim=(1, 3))
        expected = torch.full_like(per_head_max, target_mass)
        torch.testing.assert_close(per_head_max, expected, atol=1e-6, rtol=.01)
        report.update(actual_max_mass=mass.max().item(),
                      actual_mean_mass=mass.mean().item(),
                      max_logit_perturbation=(model.base.lm_head(result.hidden) - model.base.lm_head(control.hidden)).abs().max().item())
        # The calibrator's report describes the 1e-4 stage; label it explicitly.
        return {"conservative_calibration": {k: v for k, v in report.items()
                                             if k not in ("actual_max_mass", "actual_mean_mass", "max_logit_perturbation")},
                "target_mass": target_mass, "actual_max_mass": report["actual_max_mass"],
                "actual_mean_mass": report["actual_mean_mass"],
                "max_logit_perturbation": report["max_logit_perturbation"]}
    return {"target_mass": 0., "actual_max_mass": 0., "actual_mean_mass": 0., "max_logit_perturbation": 0.}


@torch.no_grad()
def measure(model, tokens, target, flags):
    model.eval()
    result = model(tokens, diagnostics=True, **flags)
    logits = model.base.lm_head(result.hidden[:, -1])
    probability = logits.softmax(-1)
    correct = (logits.argmax(-1) == target).sum().item()
    return {"examples": target.numel(), "correct": correct, "accuracy": correct / target.numel(),
            "ce": torch.nn.functional.cross_entropy(logits, target).item(),
            "mean_correct_label_probability": probability.gather(1, target[:, None]).mean().item(),
            "mean_query_memory_mass": result.diagnostics["memory_mass"][..., -1].mean().item(),
            "mean_query_null_mass": result.diagnostics["null_mass"][..., -1].mean().item()}


def evaluation(model, standard, device):
    result = {}
    policies = ({"trained_standard": {"read_memory": False, "write_memory": False, "read_null": False}}
                if standard else {"memory": {}, "zero_memory_values": {"zero_memory_values": True},
                                  "writes_disabled": {"write_memory": False},
                                  "reads_disabled": {"read_memory": False},
                                  "forget_zero": {"forget_override": 0.}})
    for delay in COMMON["evaluation_delays"]:
        generator = torch.Generator().manual_seed(8910 + delay)
        tokens, target = examples(COMMON["evaluation_examples"], delay, generator, device)
        result[str(delay)] = {name: measure(model, tokens, target, flags) for name, flags in policies.items()}
    return result


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.index < len(matrix()):
        parser.error("invalid array task index")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("training requires CUDA inside Slurm")
    frozen = json.loads((args.root / "plan.json").read_text())
    if frozen["matrix"] != matrix() or frozen["common"] != COMMON:
        raise ValueError("frozen experiment plan differs from executing code")
    arm = matrix()[args.index]
    run = args.root / "runs" / f"{arm['name']}-seed{arm['seed']}"
    run.mkdir(parents=True, exist_ok=False)
    def emit(record):
        print(json.dumps(record), flush=True)
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
    started = time.monotonic()
    model = make_model(arm["seed"], arm["input_bias"], "cuda")
    calibration = initialize_mass(model, arm["seed"], arm["mass"], "cuda")
    standard = arm["mass"] == 0
    flags = {"read_memory": False, "read_null": False, "write_memory": False} if standard else {}
    groups = model.parameter_groups("full", base_lr=COMMON["lr"], memory_lr=COMMON["lr"])
    if standard:
        for parameter in model.memory.parameters():
            parameter.requires_grad_(False)
        groups = groups[:1]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    generator = torch.Generator().manual_seed(arm["seed"])
    atomic_json(run / "config.json", {"arm": arm, "common": COMMON, "calibration": calibration,
                                      "slurm_job": os.environ["SLURM_JOB_ID"], "torch": str(torch.__version__)})
    emit({"event": "startup", "arm": arm, "calibration": calibration,
          "gpu": torch.cuda.get_device_name(), "slurm_job": os.environ["SLURM_JOB_ID"]})
    input_tokens = 0
    for step in range(COMMON["steps"] + 1):
        if step:
            delay = train_delay(arm, step)
            tokens, target = examples(COMMON["batch"], delay, generator, "cuda")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, **flags)
            logits = model.base.lm_head(output.hidden[:, -1])
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite recall language loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            input_tokens += tokens.numel()
            if step % 50 == 0:
                emit({"event": "train", "step": step, "delay": delay, "ce": loss.item(),
                      "input_tokens": input_tokens, "supervised_labels": step * COMMON["batch"],
                      "gradient_norm": norm.item(), "elapsed_seconds": time.monotonic() - started})
            del output, logits, loss
        if step in COMMON["snapshots"]:
            results = evaluation(model, standard, "cuda")
            snapshot = {"arm": arm, "step": step, "input_tokens": input_tokens,
                        "supervised_labels": step * COMMON["batch"], "results": results}
            atomic_json(run / f"step-{step:04d}.json", snapshot)
            emit({"event": "evaluation", "step": step,
                  "accuracy_delay24": next(iter(results["24"].values()))["accuracy"]})
    torch.cuda.synchronize()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "training_generator": generator.get_state(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "arm": arm, "common": COMMON,
                "step": COMMON["steps"]}, run / "final.pt")
    complete = {"status": "completed", "arm": arm, "steps": COMMON["steps"],
                "input_tokens": input_tokens, "supervised_labels": COMMON["steps"] * COMMON["batch"],
                "elapsed_seconds": time.monotonic() - started,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    atomic_json(run / "complete.json", complete)
    emit({"event": "complete", **complete})


if __name__ == "__main__":
    main()
