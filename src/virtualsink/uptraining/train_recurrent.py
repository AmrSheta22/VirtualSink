"""Slurm-only FP32 pilot trainer with document resets and explicit TBPTT state.

This is a sequential reference for correctness/profiling, not a throughput
replacement for the packed-block trainer. Corpus exhaustion never repeats data.
"""

import argparse
import inspect
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .data import sha256
from .native import load_native
from .recurrent_data import TokenDocuments
from .recurrent_sink import LayerMemoryState, MemoryState, RecurrentSinkLM


def token_losses(model, hidden, targets, chunk_size=32):
    """Chunk vocabulary projection and recompute CE activations during backward."""
    def loss_chunk(h, target):
        logits = model.base.lm_head(h)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                  target.reshape(-1), reduction="none")
    result = []
    for left in range(0, hidden.shape[1], chunk_size):
        args = (hidden[:, left:left + chunk_size], targets[:, left:left + chunk_size])
        loss = (checkpoint(loss_chunk, *args, use_reentrant=True)
                if torch.is_grad_enabled() and args[0].requires_grad else loss_chunk(*args))
        result.append(loss)
    return torch.cat(result)


def unroll_loss(model, data, cursor, state, segment_length, segments, flags, device,
                loss_chunk=32):
    """One graph spans contiguous segments; reset only at document boundaries.

    Returns a new cursor/state. Call backward once, then detach the returned
    state after the optimizer update. Batch size is deliberately one in this
    pilot, so document positions and reset semantics remain unambiguous.
    """
    document, offset = cursor
    total, count = None, 0
    for _ in range(segments):
        if document >= len(data):
            break
        inputs, targets = data.segment(document, offset, segment_length)
        result = model(inputs.to(device), state, **flags)
        losses = token_losses(model, result.hidden, targets.to(device), loss_chunk)
        summed = losses.sum()
        total = summed if total is None else total + summed
        count += losses.numel()
        offset += inputs.shape[1]
        state = result.state
        if offset == data.length(document) - 1:
            document, offset, state = document + 1, 0, None
    return total, count, state, (document, offset)


@torch.no_grad()
def evaluate(model, data, indices, segment_length, token_limit, flags, device):
    training = model.training
    model.eval()
    totals = {name: [0., 0] for name in ("all", "post_eviction", "beyond_10_windows")}
    null_mass, memory_mass, mass_count = 0., 0., 0
    try:
        for document in indices:
            state = None
            limit = min(data.length(document) - 1, token_limit)
            for start in range(0, limit, segment_length):
                inputs, targets = data.segment(document, start, min(segment_length, limit - start))
                result = model(inputs.to(device), state, diagnostics=True, **flags)
                state = result.state
                losses = token_losses(model, result.hidden, targets.to(device)).double()
                positions = torch.arange(start, start + inputs.shape[1], device=device)
                for name, minimum in (("all", 0), ("post_eviction", model.capacity),
                                      ("beyond_10_windows", 10 * model.capacity)):
                    selected = losses[positions >= minimum]
                    totals[name][0] += selected.sum().item()
                    totals[name][1] += selected.numel()
                null_mass += result.diagnostics["null_mass"].sum().item()
                memory_mass += result.diagnostics["memory_mass"].sum().item()
                mass_count += result.diagnostics["null_mass"].numel()
        report = {}
        for name, (loss, count) in totals.items():
            ce = loss / count if count else None
            report[name] = {"tokens": count, "ce": ce,
                            "ppl": math.exp(ce) if ce is not None and ce < 700 else None}
        report.update(documents=len(indices), mean_null_mass=null_mass / mass_count,
                      mean_memory_mass=memory_mass / mass_count)
        return report
    finally:
        model.train(training)


def pack_state(state):
    if state is None:
        return None
    return {"position": state.position, "capacity": state.capacity,
            "layers": [{name: getattr(layer, name).detach().cpu()
                        for name in ("key", "value", "cell", "exposed")} for layer in state.layers]}


def unpack_state(saved, device):
    if saved is None:
        return None
    layers = tuple(LayerMemoryState(*(layer[name].to(device) for name in
                                     ("key", "value", "cell", "exposed"))) for layer in saved["layers"])
    return MemoryState(layers, saved["position"], saved["capacity"])


def save_checkpoint(path, model, optimizer, scheduler, state, cursor, step, best,
                    input_tokens, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "state": pack_state(state),
                "cursor": cursor, "step": step, "best": best, "input_tokens": input_tokens,
                "config": config, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("memory", "attention", "full"), default="memory")
    parser.add_argument("--variant", choices=("memory", "standard", "null", "write_disabled", "forget_zero"), default="memory")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--timescales", type=float, nargs=3, default=(128., 512., 2048.))
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--unroll-segments", type=int, default=4)
    parser.add_argument("--loss-chunk", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--memory-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--calibration-docs", type=int, default=2)
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--initial-mass", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-docs", type=int, default=2)
    parser.add_argument("--eval-tokens", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-hours", type=float, default=1.)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for name in ("capacity", "segment_length", "unroll_segments", "loss_chunk", "steps",
                 "calibration_docs", "eval_every", "eval_docs", "eval_tokens", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.warmup < 0 or args.warmup >= args.steps:
        parser.error("warmup must lie between zero and steps-1")
    if (args.calibration_tokens <= args.capacity or args.eval_tokens <= args.capacity
            or not 0 < args.initial_mass < .1 or not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("invalid calibration/evaluation horizon, mass or time budget")
    if args.variant == "standard" and args.mode == "memory":
        parser.error("standard control needs trainable backbone projections; use --mode attention")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("training requires a working CUDA GPU inside Slurm")
    if args.output_dir.exists() and not args.resume:
        parser.error("choose a fresh output directory")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = load_native(args.model_path)
    model = RecurrentSinkLM(base, args.capacity, args.timescales).to("cuda")
    groups = model.parameter_groups(args.mode, args.base_lr, args.memory_lr)
    flags = {"read_memory": args.variant not in ("standard", "null"),
             "write_memory": args.variant not in ("standard", "null", "write_disabled"),
             "read_null": args.variant != "standard",
             "forget_override": 0. if args.variant == "forget_zero" else None}
    if args.variant in ("standard", "null"):
        for name, parameter in model.memory.named_parameters():
            parameter.requires_grad_(args.variant == "null" and (name.endswith("null_weight") or name.endswith("null_bias")))
        for group in groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        groups = [group for group in groups if group["params"]]
    datasets = {split: TokenDocuments(args.data_dir, split, base.config.vocab_size)
                for split in ("train", "validation", "test")}
    manifest = datasets["train"].manifest
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        parser.error("manifest must pin an immutable model revision")
    if len({entry["sha256"] for entry in manifest["splits"].values()}) != 3:
        parser.error("tokenized splits must differ")
    context = getattr(base.config, "max_position_embeddings", None)
    if context and any(int((data.offsets[1:] - data.offsets[:-1]).max()) - 1 > context for data in datasets.values()):
        parser.error("documents exceed pretrained position limit; prepare an explicitly truncated corpus")
    def eligible(data, count):
        selected = [i for i in range(len(data)) if data.length(i) - 1 > args.capacity][:count]
        if len(selected) != count:
            parser.error("not enough documents that exceed the cache window")
        return selected
    calibration_ids = eligible(datasets["train"], args.calibration_docs)
    validation_ids = eligible(datasets["validation"], args.eval_docs)
    test_ids = eligible(datasets["test"], args.eval_docs)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config.update(manifest=manifest, torch=str(torch.__version__), flags=flags,
                  calibration_documents=calibration_ids, validation_documents=validation_ids, test_documents=test_ids,
                  source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                 ("train_recurrent.py", "recurrent_sink.py", "recurrent_data.py", "native.py", "qwen.py", "data.py")},
                  pretrained_sha256={p.name: sha256(p) for p in sorted(args.model_path.iterdir())
                                     if p.suffix == ".safetensors" or p.name in ("config.json", "model.safetensors.index.json")})
    # Roundtrip tuples to JSON lists so resume equality is stable across serialization.
    config = json.loads(json.dumps(config))
    model.checkpointing = args.checkpointing
    optimizer = torch.optim.AdamW(groups, foreach=False)
    def schedule(step):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        return .5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step, best, input_tokens, cursor, state = 0, math.inf, 0, (0, 0), None
    calibration = None
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved["config"] != config:
            raise ValueError("resume source/config/data/pretrained mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, best, input_tokens = saved["step"], saved["best"], saved["input_tokens"]
        cursor, state = tuple(saved["cursor"]), unpack_state(saved["state"], "cuda")
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    elif args.variant != "standard":
        calibration = model.calibrate([datasets["train"].segment(i, 0, args.calibration_tokens)[0].cuda()
                                       for i in calibration_ids], target_mass=args.initial_mass, flags=flags)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    def emit(record):
        print(json.dumps(record), flush=True)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
    emit({"event": "startup", "slurm_job": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(),
          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
          "calibration": calibration, "resume_step": step})
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    started = time.monotonic()
    reason = "complete"
    while step < args.steps:
        if stopped[0] or time.monotonic() - started >= args.max_hours * 3600:
            reason = "paused"
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        total, count, state, cursor = unroll_loss(model, datasets["train"], cursor, state,
                                                 args.segment_length, args.unroll_segments,
                                                 flags, "cuda", args.loss_chunk)
        if not count:
            reason = "corpus_exhausted"
            break
        loss = total / count
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite recurrent language loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        state = state.detach() if state is not None else None
        step += 1
        input_tokens += count
        del total
        duration = time.monotonic() - tick
        emit({"event": "train", "step": step, "ce": loss.item(), "input_tokens": input_tokens,
              "cursor": cursor, "step_seconds": duration, "tokens_per_second": count / duration,
              "learning_rates": rates, "gradient_norm": norm.item(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        del loss
        if step % args.eval_every == 0 or step == args.steps:
            report = evaluate(model, datasets["validation"], validation_ids, args.segment_length,
                              args.eval_tokens, flags, "cuda")
            emit({"event": "validation", "step": step, **report})
            ce = report["post_eviction"]["ce"]
            if ce < best:
                best = ce
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler, state,
                                cursor, step, best, input_tokens, config)
        if step % args.save_every == 0:
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler, state,
                            cursor, step, best, input_tokens, config)
    save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler, state,
                    cursor, step, best, input_tokens, config)
    summary = {"status": reason, "step": step, "requested_steps": args.steps,
               "input_tokens": input_tokens, "training_elapsed_seconds": time.monotonic() - started,
               "best_validation_post_eviction_ce": best if math.isfinite(best) else None}
    # Test remains untouched during training and is reported only at completion.
    if reason == "complete":
        selected = load_checkpoint(args.output_dir / "best.pt")
        if selected["config"] != config:
            raise ValueError("selected checkpoint identity mismatch")
        model.load_state_dict(selected["model"])
        summary["selected_checkpoint_step"] = selected["step"]
        del selected
        summary["selected_checkpoint_test"] = evaluate(model, datasets["test"], test_ids,
                                                        args.segment_length, args.eval_tokens, flags, "cuda")
    summary["elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit({"event": "summary", **summary})


if __name__ == "__main__":
    main()
