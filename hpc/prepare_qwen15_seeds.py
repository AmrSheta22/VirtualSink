"""Pin larger Qwen weights and reuse token streams only after tokenizer parity."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from virtualsink.uptraining.data import sha256


def main():
    assert os.environ.get("SLURM_JOB_ID")
    root = Path(os.environ["QWEN_SEEDS_ROOT"])
    source = Path.home() / "data/phase3-k80-20260909/data/qwen05"
    original = json.loads((source / "manifest.json").read_text())
    identity_path = root / "model-identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
    else:
        identity = {"model": "Qwen/Qwen2.5-1.5B",
                    "revision": HfApi().model_info("Qwen/Qwen2.5-1.5B").sha}
        identity_path.write_text(json.dumps(identity, indent=2))
    model_path = Path(snapshot_download(identity["model"], revision=identity["revision"],
        cache_dir=str(Path.home() / "data/phase3-k80-20260909/hf-cache/hub"),
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json",
                        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))
    old_tokenizer = json.loads((Path(original["model_path"]) / "tokenizer.json").read_text())
    new_tokenizer = json.loads((model_path / "tokenizer.json").read_text())
    assert old_tokenizer == new_tokenizer, "Tokenizers differ; do not reuse token streams"
    old_config = json.loads((Path(original["model_path"]) / "config.json").read_text())
    config = json.loads((model_path / "config.json").read_text())
    assert old_config["eos_token_id"] == config["eos_token_id"]
    destination = root / "data"
    destination.mkdir(exist_ok=True)
    manifest = dict(original, model=identity["model"], revision=identity["revision"],
                    model_path=str(model_path))
    for split, spec in manifest["splits"].items():
        target = destination / spec["file"]
        if not target.exists():
            target.symlink_to(source / spec["file"])
        assert sha256(target) == spec["sha256"], split
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    identity.update(model_path=str(model_path), tokenizer_sha256=sha256(model_path / "tokenizer.json"),
                    tokenizer_exact_json_parity=True, eos_parity=True,
                    config=config, weights_sha256={p.name: sha256(p) for p in model_path.glob("*.safetensors")})
    identity_path.write_text(json.dumps(identity, indent=2))
    print(json.dumps({"event": "PREPARATION_PASSED", "model": identity["model"],
                      "revision": identity["revision"], "tokenizer_exact_json_parity": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
