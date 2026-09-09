"""Download pinned open models and tokenize a shared WikiText-103 pilot subset."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .data import sha256


MODELS = {"smollm135": "HuggingFaceTB/SmolLM-135M", "qwen05": "Qwen/Qwen2.5-0.5B"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8_192_000)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("prepare data in a CPU Slurm allocation")
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    args.output_dir.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    dataset = "Salesforce/wikitext"
    revision = api.dataset_info(dataset).sha
    files = api.list_repo_files(dataset, repo_type="dataset", revision=revision)
    tokenizer, manifests, destinations = {}, {}, {}
    for slug, name in MODELS.items():
        model_revision = api.model_info(name).sha
        model_path = snapshot_download(name, revision=model_revision,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json", "tokenizer.json",
                            "tokenizer_config.json", "special_tokens_map.json", "README.md"])
        print(f"Downloaded {name}@{model_revision} to {model_path}", flush=True)
        tokenizer[slug] = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        destination = args.output_dir / slug
        destination.mkdir()
        destinations[slug] = destination
        manifests[slug] = {"format": "little-endian uint32", "model": name,
            "revision": model_revision, "model_path": model_path,
            "source": {"dataset": dataset, "revision": revision, "configuration": "wikitext-103-raw-v1",
                       "split_policy": "official splits; same ordered source rows for both models",
                       "license_reference": f"https://huggingface.co/datasets/{dataset}/blob/{revision}/README.md"},
            "packing": "nonempty rows followed by EOS; cross-row attention; no padding", "splits": {}}
    for split in ("train", "validation", "test"):
        selected = sorted(f for f in files if f.startswith(f"wikitext-103-raw-v1/{split}-") and f.endswith(".parquet"))
        if not selected:
            raise ValueError(f"missing parquet files for {split}")
        counts = dict.fromkeys(MODELS, 0)
        handles = {slug: (destinations[slug] / f"{split}.bin").open("wb") for slug in MODELS}
        eos = {slug: json.loads((Path(manifests[slug]["model_path"]) / "config.json").read_text())["eos_token_id"] for slug in MODELS}
        source_files, rows = [], 0
        target = args.tokens if split == "train" else 131_072
        try:
            for remote in selected:
                path = hf_hub_download(dataset, remote, repo_type="dataset", revision=revision)
                source_files.append({"file": remote, "sha256": sha256(path)})
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
                    texts = [t for t in batch.column(0).to_pylist() if t and t.strip()]
                    for slug in MODELS:
                        encoded = tokenizer[slug].encode_batch(texts, add_special_tokens=False)
                        for record in encoded:
                            ids = record.ids + [eos[slug]]
                            np.asarray(ids, dtype="<u4").tofile(handles[slug])
                            counts[slug] += len(ids)
                    rows += len(texts)
                    if min(counts.values()) >= target:
                        break
                if min(counts.values()) >= target:
                    break
        finally:
            for handle in handles.values():
                handle.close()
        for slug in MODELS:
            path = destinations[slug] / f"{split}.bin"
            if counts[slug] < target:
                raise ValueError(f"insufficient unique {split} tokens for {slug}: {counts[slug]}")
            manifests[slug]["splits"][split] = {"file": path.name, "tokens": counts[slug],
                "sha256": sha256(path), "source_files": source_files, "source_rows": rows}
        print(json.dumps({"split": split, "tokens": counts, "source_rows": rows}), flush=True)
    for slug in MODELS:
        (destinations[slug] / "manifest.json").write_text(json.dumps(manifests[slug], indent=2))
    print("PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
