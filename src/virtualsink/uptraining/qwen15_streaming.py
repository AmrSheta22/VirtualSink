"""Frozen Qwen1.5B streaming replication: twelve fixed articles and five policies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from .data import sha256
from .evaluate_streaming import CONFIRMATION_TITLES, check_parity, evaluate, write
from .native import load_native
from .qwen import disable_sinks
from .rebased_reference import RebasedReference
from .streaming_fifo import StreamingDecoder

POLICIES = ["standard-fifo", "standard-prefix4", "standard-rebased-prefix4",
            "query_sink-fifo", "query_sink-disabled"]


def configs(root):
    runs = {variant: root / "seeds/seed1729/runs" / variant for variant in ["standard", "query_sink"]}
    config = {variant: json.loads((path / "config.json").read_text()) for variant, path in runs.items()}
    for key in ["manifest", "mode", "seq_len", "window_size", "steps", "batch_size", "accumulation", "seed", "precision", "backend", "chunk_size"]:
        assert config["standard"][key] == config["query_sink"][key], key
    assert config["standard"]["manifest"]["model"] == "Qwen/Qwen2.5-1.5B"
    assert all(json.loads((path / "summary.json").read_text())["best_step"] == 975 for path in runs.values())
    return runs, config


def article(manifest, index):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    cache = Path.home() / "data/phase3-k80-20260909/hf-cache/hub"
    source = next(cache.glob("datasets--Salesforce--wikitext/snapshots/" + manifest["source"]["revision"] + "/wikitext-103-raw-v1/test-*.parquet"))
    assert sha256(source) == manifest["splits"]["test"]["source_files"][0]["sha256"]
    title = CONFIRMATION_TITLES[index]
    parts, active = [], False
    for row in pq.read_table(source, columns=["text"]).column("text").to_pylist():
        if re.match(r"^\s*= [^=].*[^=] =\s*$", row):
            if active:
                break
            active = row.strip() == title
        if active:
            parts.append(row)
    text = "".join(parts)
    tokenizer = Tokenizer.from_file(str(Path(manifest["model_path"]) / "tokenizer.json"))
    encoded = tokenizer.encode(text).ids
    assert len(encoded) >= 8193
    ids = np.asarray(encoded[:8193], dtype="<i8")
    return tokenizer, ids, {"title": title, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "available_tokens": len(encoded)}


def load_checkpoint(path, config):
    model = load_native(config["manifest"]["model_path"], 32).eval()
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["step"] == 975 and checkpoint["config"] == config
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    return model.cuda()


@torch.no_grad()
def rebase_parity(model, tokens):
    assert all(layer.self_attn.sink_disabled for layer in model.model.layers)
    actual, reference = StreamingDecoder(model, 512, 4, rebase=True), RebasedReference(model, 512, 4)
    maximum = 0.
    for position, token in enumerate(tokens):
        logits, *_ = actual.step(token)
        expected = reference.step(token)
        maximum = max(maximum, (logits - expected).abs().max().item())
        torch.testing.assert_close(logits, expected, atol=.003, rtol=.0001)
        assert all(key.shape[2] <= 512 for key, value in actual.cache)
    return {"tokens": len(tokens), "post_eviction_positions": len(tokens) - 512,
            "max_absolute_logit_error": maximum, "atol": .003, "rtol": .0001,
            "reference": "absolute-rotated cached keys with delta rotation to retained cache indices"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--article-index", type=int, choices=range(12), default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("Slurm GPU allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    runs, config = configs(args.training_root)
    expected = json.loads((args.output.parent.parent / "checkpoint-identity.json").read_text())
    actual_hashes = {variant: sha256(path / "best.pt") for variant, path in runs.items()}
    assert actual_hashes == expected["checkpoint_sha256"]
    tokenizer, encoded, identity = article(config["standard"]["manifest"], args.article_index)
    tokens = torch.tensor(encoded, device="cuda")
    provenance = {"article_index": args.article_index, "article": identity,
        "training_config": config, "checkpoint_sha256": actual_hashes, "best_step": 975,
        "capacity": 512, "predictions": 8192, "policies": POLICIES,
        "source_sha256": {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")},
        "job": os.environ["SLURM_JOB_ID"], "validate_only": args.validate_only}
    write(args.output / "config.json", provenance)
    for variant in ["standard", "query_sink"]:
        model = load_checkpoint(runs[variant] / "best.pt", config[variant])
        disable_sinks(model, variant == "standard")
        checks = {str(window): check_parity(model, tokens[:641], window) for window in [None, 512]}
        if variant == "standard":
            checks["rebased_prefix"] = rebase_parity(model, tokens[:1153] if args.validate_only else tokens[:641])
        write(args.output / ("parity-" + variant + ".json"), checks)
        print("PARITY_PASSED", variant, json.dumps(checks), flush=True)
        if not args.validate_only:
            policies = [("standard-fifo", 0, False, True), ("standard-prefix4", 4, False, True),
                        ("standard-rebased-prefix4", 4, True, True)] if variant == "standard" else [
                        ("query_sink-fifo", 0, False, False), ("query_sink-disabled", 0, False, True)]
            for label, prefix, rebase, disabled in policies:
                disable_sinks(model, disabled)
                evaluate(model, tokens, tokenizer, prefix, args.output, label, rebase=rebase)
        del model
        torch.cuda.empty_cache()
    write(args.output / "complete.json", {"complete": True, "validate_only": args.validate_only,
                                          "policies": POLICIES, "article_index": args.article_index})


if __name__ == "__main__":
    main()
