"""Slurm-only Qwen2.5 continued pretraining with validation and resumable state."""

import argparse
import json
import math
import os
import signal
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import TokenBlocks, sha256
from .qwen import adapt_qwen, auxiliary_loss, parameter_groups, disable_sinks


def language_loss(model, tokens, chunk_size=128):
    # Avoid allocating B*T*vocabulary logits (Qwen vocabulary ~152k).
    result = model.model(input_ids=tokens, use_cache=False, output_attentions=True,
                         return_dict=True)
    hidden, targets = result.last_hidden_state[:, :-1], tokens[:, 1:]
    def ce(h, target):
        return F.cross_entropy(model.lm_head(h).float().reshape(-1, model.config.vocab_size),
                               target.reshape(-1), reduction="sum")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        args = (hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size])
        total = total + (checkpoint(ce, *args, use_reentrant=False)
                         if torch.is_grad_enabled() else ce(*args))
    return total / targets.numel(), result.attentions


@torch.no_grad()
def evaluate(model, data, batch_size, batches, device, precision="bf16"):
    model.eval()
    total, count = 0.0, 0
    gate_sum = None
    for start in range(0, min(len(data), batch_size * batches), batch_size):
        tokens = torch.stack([data[i] for i in range(start, min(start + batch_size, len(data)))]).to(device)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()):
            loss, gates = language_loss(model, tokens)
        n = tokens.shape[0]
        total += loss.item() * n
        count += n
        means = torch.stack([g.float().mean((0, 2)) for g in gates])
        gate_sum = means * n if gate_sum is None else gate_sum + means * n
    means = gate_sum / count
    return {"ce": total / count, "perplexity": math.exp(min(total / count, 700)),
            "sequences": count, "physical_gate_by_layer_head": means.cpu().tolist(),
            "heads_below_0.01": int((means < 0.01).sum())}


def save_state(path, model, optimizer, scheduler, step, best, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "best": best,
                "config": config, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("attention", "full"), default="attention")
    parser.add_argument("--variant", choices=("query_sink", "standard"), default="query_sink")
    parser.add_argument("--auxiliary", choices=("anti_collapse", "supplied", "none"), default="anti_collapse")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--backend", choices=("transformers", "native"), default="transformers")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-seconds", type=int, default=600)
    parser.add_argument("--milestone-every", type=int, default=250)
    parser.add_argument("--max-hours", type=float, default=23.5)
    parser.add_argument("--window-size", type=int, default=0, help="0: full causal attention; positive: includes current token")
    args = parser.parse_args()
    for name in ("steps", "seq_len", "batch_size", "accumulation", "chunk_size", "eval_every", "eval_batches", "save_every", "save_seconds", "milestone_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.seq_len < 2 or not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error("invalid sequence length or gamma")
    if args.window_size < 0 or args.window_size > args.seq_len:
        parser.error("window-size must be between 0 and seq-len")
    if args.window_size and args.backend != "native":
        parser.error("SWA currently requires the native backend")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("training must run inside a Slurm allocation")
    if not torch.cuda.is_available():
        parser.error("a working CUDA GPU is required")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 requires a newer GPU; use --backend native --precision fp32 on K80")
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("max-hours must be positive")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        parser.error("manifest must pin an immutable model revision")
    if manifest.get("format") != "little-endian uint32":
        parser.error("unsupported token format")
    if len({manifest["splits"][s]["sha256"] for s in ("train", "validation", "test")}) != 3:
        parser.error("train/validation/test token streams must differ")
    if manifest["model"] not in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM-135M"):
        parser.error("use a supported base model")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config.update(manifest=manifest, torch=str(torch.__version__))
    config["source_sha256"] = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    if args.output_dir.exists() and not args.resume:
        parser.error("choose a fresh output directory")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # FP32 master weights/Adam state with BF16 operations; budget these honestly.
    if args.backend == "native":
        if args.precision != "fp32":
            parser.error("native backend is validated for FP32 only")
        from .native import load_native, set_window
        model = load_native(manifest["model_path"], args.chunk_size)
        set_window(model, args.window_size or None)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["revision"],
                                                    torch_dtype=torch.float32, attn_implementation="eager")
        adapt_qwen(model, args.chunk_size)
    if args.seq_len > model.config.max_position_embeddings:
        parser.error("sequence length exceeds model context")
    datasets = {}
    for split in ("train", "validation", "test"):
        entry = manifest["splits"][split]
        path = args.data_dir / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{split} token checksum mismatch")
        datasets[split] = TokenBlocks(path, args.seq_len, model.config.vocab_size)
        if datasets[split].tokens.size != entry["tokens"]:
            raise ValueError(f"{split} token count does not match manifest")
    required = args.steps * args.batch_size * args.accumulation
    if len(datasets["train"]) < required:
        parser.error(f"need {required * args.seq_len:,} training tokens; refusing silent corpus repetition")
    model.to("cuda")
    groups = parameter_groups(model, args.mode)
    if args.variant == "standard":
        disable_sinks(model)
        for param in groups[1]["params"]:
            param.requires_grad_(False)
        groups = groups[:1]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    warmup = min(100, args.steps // 10)
    def lr_scale(current):
        if current < warmup:
            return current / max(1, warmup)
        return max(0., .5 * (1 + math.cos(math.pi * (current - warmup) / max(1, args.steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step, best = 0, math.inf
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        if saved["config"] != config:
            raise ValueError("resume config/source/data mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, best = saved["step"], saved["best"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "gpu": torch.cuda.get_device_name(), "input_tokens": required * args.seq_len,
                      "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                      "slurm_job": os.environ.get("SLURM_JOB_ID")}), flush=True)
    started = time.monotonic()
    last_save = started
    step_seconds, evaluation_seconds, checkpoint_seconds = [], [], []
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    def validate(data):
        before = time.monotonic()
        result = evaluate(model, data, args.batch_size, args.eval_batches, "cuda", args.precision)
        evaluation_seconds.append(time.monotonic() - before)
        return result
    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        def emit(row):
            row["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            text = json.dumps(row, allow_nan=False)
            log.write(text + "\n")
            log.flush()
            print(text, flush=True)
        def snapshot(name, rotate=False):
            nonlocal last_save
            before = time.monotonic()
            target = args.output_dir / name
            # Keep previous last intact until the new file is fully written.
            staging = args.output_dir / "pending.pt"
            save_state(staging, model, optimizer, scheduler, step, best, config)
            if rotate and target.exists():
                target.replace(args.output_dir / "previous.pt")
            staging.replace(target)
            last_save = time.monotonic()
            duration = last_save - before
            checkpoint_seconds.append(duration)
            emit({"step": step, "checkpoint": name, "bytes": target.stat().st_size,
                  "checkpoint_seconds": duration})
        if step == 0:
            disable_sinks(model)
            baseline = validate(datasets["validation"])
            emit({"step": 0, "pretrained_validation": baseline})
            disable_sinks(model, args.variant == "standard")
            initial = validate(datasets["validation"])
            best = initial["ce"]
            emit({"step": 0, "adapted_initial_validation": initial})
            snapshot("best.pt")
            snapshot("last.pt", rotate=True)
        while step < args.steps:
            step_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lm_total, aux_total = 0., 0.
            for micro in range(args.accumulation):
                start = (step * args.accumulation + micro) * args.batch_size
                tokens = torch.stack([datasets["train"][i] for i in range(start, start + args.batch_size)]).cuda()
                with (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()):
                    lm, gates = language_loss(model, tokens, args.chunk_size)
                    aux, means = auxiliary_loss(gates, args.auxiliary if args.variant != "standard" else "none")
                    loss = (lm + args.gamma * aux) / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                lm_total += lm.item() / args.accumulation
                aux_total += aux.item() / args.accumulation
                if micro % 8 == 0:
                    emit({"step": step, "microbatch": micro + 1, "microbatch_lm_ce": lm.item()})
                del loss, lm, aux, gates, means
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            torch.cuda.synchronize()
            duration = time.monotonic() - step_started
            step_seconds.append(duration)
            emit({"step": step, "lm_ce": lm_total, "auxiliary": aux_total,
                  "grad_norm": norm.item(), "elapsed_seconds": time.monotonic() - started,
                  "step_seconds": duration, "tokens_per_second": args.seq_len * args.batch_size * args.accumulation / duration,
                  "learning_rates": [g["lr"] for g in optimizer.param_groups],
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            if len(step_seconds) >= 5:
                timing = estimate_remaining(step_seconds[2:], args.steps - step, args.eval_every,
                                            args.save_every, evaluation_seconds, checkpoint_seconds,
                                            args.save_seconds, args.milestone_every)
                timing.update(step=step, excludes="queue, downloads, initial setup", measured_steps=len(step_seconds) - 2)
                (args.output_dir / "timing.json").write_text(json.dumps(timing, indent=2))
                if step % 10 == 0 or step == 5:
                    emit({"step": step, "timing": timing})
            if step % args.eval_every == 0 or step == args.steps:
                metrics = validate(datasets["validation"])
                emit({"step": step, "validation": metrics})
                if metrics["ce"] < best:
                    best = metrics["ce"]
                    snapshot("best.pt")
            if step % args.milestone_every == 0:
                snapshot(f"step-{step:06d}.pt")
            stopping = stop["requested"] or time.monotonic() - started >= args.max_hours * 3600
            if (step % args.save_every == 0 or time.monotonic() - last_save >= args.save_seconds
                    or stopping or step == args.steps):
                snapshot("last.pt", rotate=True)
            if stopping and step < args.steps:
                status = {"status": "paused", "step": step, "reason": "signal or wall time budget", "resume": str(args.output_dir / "last.pt")}
                (args.output_dir / "status.json").write_text(json.dumps(status, indent=2))
                emit(status)
                return
        saved = torch.load(args.output_dir / "best.pt", map_location="cpu")
        model.load_state_dict(saved["model"])
        best_step = saved["step"]
        del saved
        test = validate(datasets["test"])
        (args.output_dir / "summary.json").write_text(json.dumps({"best_step": best_step, "test": test}, indent=2))
        emit({"status": "complete", "best_step": best_step, "test": test})


def estimate_remaining(steps, remaining, eval_every, save_every, evaluations, saves,
                       save_seconds=600, milestone_every=250):
    """Observed throughput plus measured overhead; range is descriptive, not a CI."""
    mean = statistics.mean(steps)
    low, high = min(steps), max(steps)
    # Include final test, validation, periodic last, possible best, and milestones.
    effective_save_interval = max(1, min(save_every, save_seconds / mean))
    overhead = ((math.ceil(remaining / eval_every) + 1) * statistics.mean(evaluations or [0])
                + (math.ceil(remaining / effective_save_interval) + math.ceil(remaining / eval_every)
                   + math.ceil(remaining / milestone_every)) * statistics.mean(saves or [0]))
    return {"mean_step_seconds": mean, "remaining_seconds": remaining * mean + overhead,
            "remaining_observed_range_seconds": [remaining * low + overhead, remaining * high + overhead],
            "estimated_overhead_seconds": overhead}


if __name__ == "__main__":
    main()
