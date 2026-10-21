"""Direct memory supervision under conservative joint attention initialization.

One-layer toy experiment only. Auxiliary decoding bypasses physical/null
normalization and reuses existing O projection, final norm and LM head. No new
decoder parameters or pretrained-Qwen quality claims are introduced.
"""

import argparse
import json
import os
import time

import torch

from .recall_experiment import SEEDS, atomic_json, examples, initialize_mass, make_model


ARMS = (
    dict(name="normal_full", warmup="full_normal", retain_aux=False),
    dict(name="frozen_normal", warmup="frozen_normal", retain_aux=False),
    dict(name="aux_then_normal", warmup="frozen_aux", retain_aux=False),
    dict(name="aux_then_joint", warmup="frozen_aux", retain_aux=True),
)
COMMON = dict(steps=1300, warmup_steps=300, adaptation_steps=1000, snapshots=[0, 300, 1300],
              batch=32, capacity=8, timescales=[128., 512., 2048.], lr=.003,
              train_delay=24, evaluation_delays=[24, 48, 96], evaluation_examples=512,
              initial_mass=1e-4, input_bias=-4., auxiliary_weight=1.)


def matrix():
    return [dict(index=i * len(ARMS) + j, seed=seed, **arm)
            for i, seed in enumerate(SEEDS) for j, arm in enumerate(ARMS)]


def auxiliary_logits(model, tokens, state):
    """Read the final memory without multiplying by its tiny joint softmax mass.

    For a one-layer model the final raw query is the projected input embedding;
    this is exactly the query used by the memory branch in the causal forward.
    Auxiliary decoding omits local attention, query residual and MLP. It uses
    only existing projections, with no task-specific decoder parameters.
    """
    if len(model.memory) != 1:
        raise ValueError("auxiliary reference only supports the one-layer toy model")
    layer, bank = model.base.model.layers[0], model.memory[0]
    attention = layer.self_attn
    hidden = layer.input_layernorm(model.base.model.embed_tokens(tokens[:, -1:]))
    query = attention.q_proj(hidden).view(tokens.shape[0], attention.heads, attention.head_dim)
    _, scores, values = bank.read(query, query, state.layers[0].exposed)
    read = torch.einsum("bhs,bhsd->bhd", scores.softmax(-1), values)
    projected = attention.o_proj(read.reshape(tokens.shape[0], -1))
    return model.base.lm_head(model.base.model.norm(projected))


def optimizer_for_phase(model, arm, adapting=False, previous=None):
    frozen = not adapting and arm["warmup"] != "full_normal"
    groups = model.parameter_groups("memory" if frozen else "full", COMMON["lr"], COMMON["lr"])
    if frozen:
        # Frozen read keys bound content scores while cells/values learn. Absolute
        # score biases and null branch stay fixed until normal adaptation.
        for name, parameter in model.memory.named_parameters():
            if name.rsplit(".", 1)[-1] in ("read_key", "null_weight", "null_bias", "memory_bias"):
                parameter.requires_grad_(False)
        groups[0]["params"] = [p for p in groups[0]["params"] if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    if previous is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter in previous.state:
                    optimizer.state[parameter] = previous.state[parameter]
    return optimizer


@torch.no_grad()
def evaluation(model, device):
    model.eval()
    results = {}
    for delay in COMMON["evaluation_delays"]:
        generator = torch.Generator().manual_seed(8910 + delay)
        tokens, target = examples(COMMON["evaluation_examples"], delay, generator, device)
        policies = {"joint": {}, "writes_disabled": {"write_memory": False},
                    "reads_disabled": {"read_memory": False},
                    "zero_memory_values": {"zero_memory_values": True},
                    "forget_zero": {"forget_override": 0.}}
        measures = {}
        for name, flags in policies.items():
            output = model(tokens, diagnostics=True, **flags)
            logits = model.base.lm_head(output.hidden[:, -1])
            correct = (logits.argmax(-1) == target).sum().item()
            measures[name] = {"examples": len(target), "correct": correct,
                              "accuracy": correct / len(target),
                              "ce": torch.nn.functional.cross_entropy(logits, target).item(),
                              "query_memory_mass": output.diagnostics["memory_mass"][..., -1].mean().item()}
            if name in ("joint", "writes_disabled", "forget_zero"):
                logits = auxiliary_logits(model, tokens, output.state)
                correct = (logits.argmax(-1) == target).sum().item()
                measures["aux_" + name] = {"examples": len(target), "correct": correct,
                                         "accuracy": correct / len(target),
                                         "ce": torch.nn.functional.cross_entropy(logits, target).item()}
        results[str(delay)] = measures
    return results


@torch.no_grad()
def probe(model, tokens, original_logits):
    model.eval()
    result = model(tokens, diagnostics=True)
    mass = result.diagnostics["null_mass"] + result.diagnostics["memory_mass"]
    current_base = model(tokens, read_memory=False, read_null=False)
    logits = model.base.lm_head(result.hidden)
    return {"max_virtual_mass": mass.max().item(), "mean_virtual_mass": mass.mean().item(),
            "max_joint_logit_drift_from_original_base": (logits - original_logits).abs().max().item(),
            "max_memory_logit_contribution": (logits - model.base.lm_head(current_base.hidden)).abs().max().item()}


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.index < len(matrix()):
        parser.error("invalid task index")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("training requires Slurm/CUDA")
    plan = json.loads((args.root / "plan.json").read_text())
    if plan["matrix"] != matrix() or plan["common"] != COMMON:
        raise ValueError("frozen plan mismatch")
    arm = matrix()[args.index]
    run = args.root / "runs" / f"{arm['name']}-seed{arm['seed']}"
    run.mkdir(parents=True, exist_ok=False)
    def emit(event):
        print(json.dumps(event), flush=True)
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    started = time.monotonic()
    model = make_model(arm["seed"], COMMON["input_bias"], "cuda")
    original_parameters = {name: p.detach().clone() for name, p in model.base.named_parameters()}
    generator_probe = torch.Generator().manual_seed(arm["seed"] + 60000)
    probe_tokens, _ = examples(64, 24, generator_probe, "cuda")
    with torch.no_grad():
        original_logits = model.base.lm_head(model(probe_tokens, read_memory=False, read_null=False).hidden)
    calibration = initialize_mass(model, arm["seed"], COMMON["initial_mass"], "cuda")
    optimizer = optimizer_for_phase(model, arm)
    generator = torch.Generator().manual_seed(arm["seed"])
    atomic_json(run / "config.json", {"arm": arm, "common": COMMON, "calibration": calibration,
                                      "torch": str(torch.__version__), "slurm_job": os.environ["SLURM_JOB_ID"]})
    emit({"event": "startup", "arm": arm, "calibration": calibration,
          "gpu": torch.cuda.get_device_name(), "slurm_job": os.environ["SLURM_JOB_ID"]})
    input_tokens = 0
    for step in range(COMMON["steps"] + 1):
        if step:
            tokens, target = examples(COMMON["batch"], COMMON["train_delay"], generator, "cuda")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens)
            joint_logits = model.base.lm_head(output.hidden[:, -1])
            joint_loss = torch.nn.functional.cross_entropy(joint_logits, target)
            warming = step <= COMMON["warmup_steps"]
            use_aux = (warming and arm["warmup"] == "frozen_aux") or (not warming and arm["retain_aux"])
            auxiliary_loss = None
            if use_aux:
                auxiliary_loss = torch.nn.functional.cross_entropy(auxiliary_logits(model, tokens, output.state), target)
            loss = (auxiliary_loss if warming and use_aux else joint_loss +
                    (COMMON["auxiliary_weight"] * auxiliary_loss if use_aux else 0))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite memory warmup objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            input_tokens += tokens.numel()
            if step % 50 == 0:
                emit({"event": "train", "step": step, "phase": "warmup" if warming else "adaptation",
                      "joint_ce": joint_loss.item(), "aux_ce": auxiliary_loss.item() if use_aux else None,
                      "input_tokens": input_tokens, "gradient_norm": norm.item(),
                      "elapsed_seconds": time.monotonic() - started})
            del output, joint_logits, joint_loss, auxiliary_loss, loss
        if step == COMMON["warmup_steps"]:
            before = {"results": evaluation(model, "cuda"), "probe": probe(model, probe_tokens, original_logits)}
            unchanged = all(torch.equal(p, original_parameters[name]) for name, p in model.base.named_parameters())
            if arm["warmup"] != "full_normal" and not unchanged:
                raise AssertionError("warmup changed frozen backbone parameters")
            transfer_calibration = initialize_mass(model, arm["seed"] + 10000, COMMON["initial_mass"], "cuda")
            after = probe(model, probe_tokens, original_logits)
            transfer = {"arm": arm, "base_unchanged": unchanged, "before": before,
                        "after_probe": after, "calibration": transfer_calibration}
            atomic_json(run / "transfer.json", transfer)
            optimizer = optimizer_for_phase(model, arm, adapting=True, previous=optimizer)
            emit({"event": "transfer", "base_unchanged": unchanged, "probe": after})
        if step in COMMON["snapshots"]:
            results = evaluation(model, "cuda")
            atomic_json(run / f"step-{step:04d}.json", {"arm": arm, "step": step,
                        "input_tokens": input_tokens, "supervised_labels": step * COMMON["batch"], "results": results})
            emit({"event": "evaluation", "step": step, "joint_accuracy": results["24"]["joint"]["accuracy"],
                  "aux_accuracy": results["24"]["aux_joint"]["accuracy"]})
    torch.cuda.synchronize()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(), "arm": arm, "common": COMMON}, run / "final.pt")
    complete = {"status": "completed", "arm": arm, "steps": COMMON["steps"],
                "input_tokens": input_tokens, "supervised_labels": COMMON["steps"] * COMMON["batch"],
                "elapsed_seconds": time.monotonic() - started}
    atomic_json(run / "complete.json", complete)
    emit({"event": "complete", **complete})


if __name__ == "__main__":
    main()
