"""Slurm-only paired SmolLM1 pretraining; never loads pretrained weights.

Uses the recipe's global batch (512x2048), AdamW and initialization in FP32 on
K80. Checkpoints include partial accumulation, so long steps survive requeues.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..uptraining import native
from ..uptraining.data import sha256
from ..uptraining.qwen import disable_sinks
from .data import MixedBlocks
from .recipe import initialize, shared_hash, optimizer_groups, schedule, RECIPE_URL, MODEL_CONFIG


def language_loss(model, tokens, chunk_size=64):
    result = model.model(tokens[:, :-1], use_cache=False)
    hidden, targets = result.last_hidden_state, tokens[:, 1:]
    def ce(h, y):
        return F.cross_entropy(model.lm_head(h).float().reshape(-1, model.config.vocab_size),
                               y.reshape(-1), reduction="sum")
    loss = hidden.new_zeros(())
    for start in range(0, hidden.shape[1], chunk_size):
        args = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
        loss = loss + (checkpoint(ce, *args, use_reentrant=True) if torch.is_grad_enabled() else ce(*args))
    return loss / targets.numel(), result.attentions


@torch.no_grad()
def evaluate(model, data, count, device="cuda"):
    model.eval()
    losses, gate_total = [], None
    for index in range(min(count, len(data))):
        loss, gates = language_loss(model, data[index][None].to(device))
        losses.append(loss.item())
        means = torch.stack([g.mean((0, 2)) for g in gates])
        gate_total = means if gate_total is None else gate_total + means
    ce = sum(losses) / len(losses)
    if not math.isfinite(ce):
        raise FloatingPointError("nonfinite evaluation")
    return dict(ce=ce, perplexity=math.exp(min(ce, 700)), per_block_ce=losses,
                physical_gate_by_layer_head=(gate_total / len(losses)).cpu().tolist())


@torch.no_grad()
def measure_attention(model, data, count=4, device="cuda"):
    """Joint physical/virtual mass; no physical renormalization.

Late-query diagnostics are supplemental; shared held-out CE selects checkpoints.
"""
    model.eval()
    original = native.sink_attention
    sums, calls = {}, 0
    layers = len(model.model.layers)
    def measured(q, k, v, w, b, chunk_size=64, recompute=False, window_size=None):
        nonlocal calls
        layer = calls % layers
        length = q.shape[2]
        start_late = length // 2
        total = q.new_zeros((q.shape[1], 7))
        keys = torch.arange(length, device=q.device)
        for start in range(start_late, length, chunk_size):
            qc = q[:, :, start:start + chunk_size].float()
            positions = torch.arange(start, min(start + chunk_size, length), device=q.device)
            scores = (qc @ k.float().transpose(-1, -2)) * q.shape[-1] ** -.5
            scores.masked_fill_(keys[None, :] > positions[:, None], -torch.inf)
            oldest = (positions - window_size + 1).clamp_min(0) if window_size else torch.zeros_like(positions)
            if window_size is not None:
                scores.masked_fill_(keys[None, :] < oldest[:, None], -torch.inf)
            sink = (qc * w[None, :, None]).sum(-1) + b[None, :, None]
            weights = torch.cat((scores, sink[..., None]), -1).softmax(-1)
            margin = sink - torch.logsumexp(scores, -1)
            physical = weights[..., :-1]
            oldest_mass = (physical * (keys[None, :] == oldest[:, None])).sum(-1)
            oldest4 = (physical * ((keys[None, :] >= oldest[:, None]) & (keys[None, :] < oldest[:, None] + 4))).sum(-1)
            stats = torch.stack((weights[..., 0], weights[..., :4].sum(-1),
                                 weights[..., -1], margin, (weights[..., :4].sum(-1) > .5).float(), oldest_mass, oldest4), -1)
            total += stats.sum((0, 2))
        sums[layer] = sums.get(layer, 0) + total / (q.shape[0] * (length - start_late))
        calls += 1
        return original(q, k, v, w, b, chunk_size, False, window_size)
    native.sink_attention = measured
    try:
        for index in range(min(count, len(data))):
            model.model(data[index][None, :-1].to(device))
    finally:
        native.sink_attention = original
    blocks = min(count, len(data))
    return dict(blocks=blocks, late_query_start=data.length // 2,
        columns=["first_token_mass", "first4_mass", "virtual_mass", "virtual_minus_physical_lse", "first4_majority_fraction", "oldest_visible_mass", "oldest4_visible_mass"],
        by_layer_head=torch.stack([sums[i] / blocks for i in range(layers)]).cpu().tolist())


def save(path, model, optimizer, scheduler, progress, config):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), progress=dict(progress), config=config,
        gradients={n: p.grad for n, p in model.named_parameters() if p.grad is not None},
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, model, optimizer, scheduler, config):
    saved = torch.load(path, map_location="cpu")
    if saved["config"] != config:
        raise ValueError("resume config/data/source mismatch")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        if name in saved["gradients"]:
            param.grad = saved["gradients"][name].to(param.device)
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["progress"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["standard", "query_sink"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--window-size", type=int, default=512, help="0 selects full causal attention")
    parser.add_argument("--sink-weight-std", type=float, default=.02)
    parser.add_argument("--sink-bias", type=float, default=4.)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--save-seconds", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=23.)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("a working Slurm GPU allocation is required")
    if args.sequence_length < 1 or 1048576 % args.sequence_length:
        parser.error("sequence length must divide 1048576")
    global_batch = 1048576 // args.sequence_length
    if args.steps < 2 or args.microbatch < 1 or global_batch % args.microbatch:
        parser.error("steps >=2; microbatch must divide global batch")
    if not 0 <= args.window_size <= args.sequence_length or not math.isfinite(args.sink_bias) or not math.isfinite(args.sink_weight_std) or args.sink_weight_std <= 0:
        parser.error("invalid window or sink initialization")
    if min(args.chunk_size, args.eval_every, args.eval_blocks, args.save_seconds, args.max_hours) <= 0:
        parser.error("invalid interval")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if manifest["steps"] != args.steps or manifest["sequence_length"] != args.sequence_length or manifest["global_batch_sequences"] != global_batch:
        parser.error("data budget/context mismatch")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(recipe=RECIPE_URL, manifest_sha256=sha256(args.data_dir / "manifest.json"),
        model_seed=8, data_seed=42, initialization="random Nanotron standard; no pretrained weights",
        global_batch=global_batch, sequence_length=args.sequence_length, auxiliary="none", gamma=0., precision="fp32",
        lr=.003, betas=[.9, .95], epsilon=1e-8, weight_decay=.01, clip_grad=1.,
        torch=str(torch.__version__), source_sha256={str(p.relative_to(Path(__file__).parents[1])): sha256(p)
          for folder in (Path(__file__).parent, Path(__file__).parents[1] / "uptraining") for p in folder.glob("*.py")})
    root = args.output_dir
    if (root / "summary.json").exists():
        print("Training already complete", flush=True)
        return
    if root.exists() and not (root / "last.pt").exists():
        parser.error("incomplete directory without resumable checkpoint")
    model = initialize(8, args.chunk_size, dict(MODEL_CONFIG, max_position_embeddings=args.sequence_length),
                       sink_weight_std=args.sink_weight_std, sink_bias=args.sink_bias)
    native.set_window(model, args.window_size or None)
    config["shared_initial_weights_sha256"] = shared_hash(model)
    print(json.dumps(dict(event="initialization", shared_sha256=config["shared_initial_weights_sha256"],
        parameters=sum(p.numel() for p in model.parameters()), gpu=torch.cuda.get_device_name())), flush=True)
    disable_sinks(model, args.variant == "standard")
    model.cuda()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.variant), lr=.003,
                                 betas=(.9, .95), eps=1e-8, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, args.steps))
    data = {split: MixedBlocks(args.data_dir, split) for split in ("train", "validation", "test")}
    progress = dict(step=0, micro=0, lm_sum=0., best=math.inf, training_seconds=0.)
    resumed = (root / "last.pt").exists()
    if resumed:
        progress = restore(root / "last.pt", model, optimizer, scheduler, config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    started = last_save = time.monotonic()
    accumulation = global_batch // args.microbatch
    with (root / "metrics.jsonl").open("a", encoding="utf8") as log:
        def emit(row):
            row.update(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), job=os.environ["SLURM_JOB_ID"])
            message = json.dumps(row, allow_nan=False)
            log.write(message + "\n"); log.flush(); print(message, flush=True)
        def snapshot(name):
            nonlocal last_save
            save(root / name, model, optimizer, scheduler, progress, config)
            last_save = time.monotonic()
            emit(dict(event="checkpoint", file=name, step=progress["step"], micro=progress["micro"]))
        if not resumed:
            metrics = evaluate(model, data["validation"], args.eval_blocks)
            progress["best"] = metrics["ce"]
            emit(dict(step=0, validation=metrics, attention=measure_attention(model, data["validation"])))
            snapshot("best.pt"); snapshot("last.pt")
            del metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        emit(dict(event="start_or_resume", step=progress["step"], micro=progress["micro"],
                  accumulation=accumulation, input_tokens=args.steps * global_batch * args.sequence_length))
        while progress["step"] < args.steps:
            model.train()
            if progress["micro"] == 0:
                optimizer.zero_grad(set_to_none=True)
            for micro in range(progress["micro"], accumulation):
                before = time.monotonic()
                offset = progress["step"] * global_batch + micro * args.microbatch
                tokens = torch.stack([data["train"][i] for i in range(offset, offset + args.microbatch)]).cuda()
                lm, gates = language_loss(model, tokens, args.chunk_size)
                if not torch.isfinite(lm):
                    raise FloatingPointError("nonfinite training CE")
                (lm / accumulation).backward()
                progress["lm_sum"] += lm.item() / accumulation
                progress["micro"] = micro + 1
                torch.cuda.synchronize()
                duration = time.monotonic() - before
                progress["training_seconds"] += duration
                if micro % 8 == 0 or micro + 1 == accumulation:
                    completed = progress["step"] * accumulation + micro + 1
                    emit(dict(step=progress["step"], micro=micro + 1, micro_ce=lm.item(),
                        micro_seconds=duration, estimated_remaining_training_hours=
                        progress["training_seconds"] / completed * (args.steps * accumulation - completed) / 3600,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                del lm, gates, tokens
                stopping = stop["requested"] or time.monotonic() - started > args.max_hours * 3600
                if time.monotonic() - last_save >= args.save_seconds or stopping:
                    snapshot("last.pt")
                if stopping:
                    (root / "status.json").write_text(json.dumps(dict(status="paused", **progress)))
                    emit(dict(event="paused", step=progress["step"], micro=progress["micro"]))
                    raise SystemExit(75)  # launcher requeues; never releases afterok dependencies
            sink_grad_norm = math.sqrt(sum(float(p.grad.detach().square().sum())
                for n, p in model.named_parameters() if n.endswith(("sink_w", "sink_b")) and p.grad is not None))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step(); scheduler.step()
            progress["step"] += 1
            emit(dict(step=progress["step"], lm_ce=progress["lm_sum"], grad_norm=norm.item(),
                      sink_grad_norm=sink_grad_norm, lr=lr))
            progress.update(micro=0, lm_sum=0.)
            optimizer.zero_grad(set_to_none=True)
            if progress["step"] % args.eval_every == 0 or progress["step"] == args.steps:
                metrics = evaluate(model, data["validation"], args.eval_blocks)
                emit(dict(step=progress["step"], validation=metrics,
                          attention=measure_attention(model, data["validation"])))
                if metrics["ce"] < progress["best"]:
                    progress["best"] = metrics["ce"]
                    snapshot("best.pt")
                del metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            snapshot("last.pt")
        # Fixed final step is the primary paired comparison. Best is retained
        # separately so selection cannot masquerade as matched training time.
        snapshot("final.pt")
        final = evaluate(model, data["test"], len(data["test"]))
        attention = measure_attention(model, data["test"], 8)
        summary = dict(step=progress["step"], final_test=final, attention=attention)
        if args.variant == "query_sink":
            disable_sinks(model)
            disabled = evaluate(model, data["test"], len(data["test"]))
            summary["final_test_sinks_disabled"] = disabled
            summary["disabled_minus_enabled_ce"] = disabled["ce"] - final["ce"]
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        (root / "status.json").write_text(json.dumps(dict(status="complete", step=progress["step"])))
        emit(dict(event="complete", step=progress["step"], test_ce=final["ce"]))


if __name__ == "__main__":
    main()
