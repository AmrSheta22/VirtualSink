"""Pin, tokenize and mix a reduced public reconstruction of SmolLM1 data.

Runs only in a CPU Slurm allocation. Data preparation must complete successfully
before either training arm can start. Both arms read the identical saved order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..uptraining.data import sha256
from .recipe import WEIGHTS, mixture_order, RECIPE_URL

SOURCES = {
    "fineweb": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", "fineweb-edu-dedup/"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", "cosmopedia-v2/"),
    "python": ("Avelina/python-edu-cleaned", "3fedf55e02e72dd1fd68963f991f5e91adcb0f8d", "data/"),
    "math": ("open-web-math/open-web-math", "fde8ef8de2300f5e778f56261843dab89f230815", "data/"),
    "stackoverflow": ("bigcode/stackoverflow-clean", "26b73eba12b03e382e305003c7dd176c25adc8b7", "data/"),
}


def document_split(text):
    # Same text always goes to the same split, including across corpus sources.
    bucket = int.from_bytes(hashlib.sha256(text.strip().encode()).digest()[:8], "big") % 1000
    return "validation" if bucket < 10 else "test" if bucket < 20 else "train"


class MixedBlocks:
    def __init__(self, directory, split, verify=True):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.length = manifest["sequence_length"]
        self.order = np.load(self.directory / f"{split}-order.npy", allow_pickle=False)
        if verify and sha256(self.directory / f"{split}-order.npy") != manifest["splits"][split]["order_sha256"]:
            raise ValueError("mixture order checksum mismatch")
        self.streams = []
        for name in WEIGHTS:
            entry = manifest["sources"][name]["splits"][split]
            path = self.directory / entry["file"]
            if verify and sha256(path) != entry["sha256"]:
                raise ValueError(f"token checksum mismatch: {path}")
            self.streams.append(np.memmap(path, dtype="<u4", mode="r"))
        self.offsets = np.empty(len(self.order), dtype=np.int64)
        counts = np.zeros(len(WEIGHTS), dtype=np.int64)
        for i, source in enumerate(self.order):
            self.offsets[i] = counts[source] * self.length
            counts[source] += 1
        for count, stream in zip(counts, self.streams):
            if count and stream.size < count * self.length + 1:
                raise ValueError("insufficient tokens for shifted targets")

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = self.offsets[index]
        tokens = self.streams[self.order[index]][start:start + self.length + 1]
        return torch.from_numpy(tokens.astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--allow-stackoverflow-substitution", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("prepare inside a CPU Slurm allocation")
    if args.steps < 2 or not args.allow_stackoverflow_substitution:
        parser.error("steps >=2 and explicit public StackOverflow substitution are required")
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    root = args.output_dir
    if args.sequence_length < 1 or 1048576 % args.sequence_length:
        parser.error("sequence length must divide 1048576")
    global_batch = 1048576 // args.sequence_length
    heldout_blocks = max(128, 1048576 // args.sequence_length)
    if (root / "manifest.json").exists():
        saved = json.loads((root / "manifest.json").read_text())
        expected = dict(steps=args.steps, sequence_length=args.sequence_length,
                        global_batch_sequences=global_batch, tokenizer_revision=args.tokenizer_revision)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing data does not match requested preparation settings")
        if any(saved["splits"][s]["blocks"] != heldout_blocks for s in ("validation", "test")):
            raise ValueError("Existing held-out token budget differs")
        for split in ("train", "validation", "test"):
            MixedBlocks(root, split)
        print(f"Data already fully prepared at {root}; skipping preparation.", flush=True)
        return
    root.mkdir(parents=True, exist_ok=False)
    tokenizer_path = hf_hub_download("HuggingFaceTB/cosmo2-tokenizer", "tokenizer.json", revision=args.tokenizer_revision)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # The recipe specifies BOS=EOS=0. Validate the actual public tokenizer.
    if tokenizer.token_to_id("<|endoftext|>") != 0 or tokenizer.get_vocab_size() != 49152:
        raise ValueError("unexpected tokenizer special token/vocabulary")
    quotas, split_metadata = {}, {}
    for split, count in (("train", args.steps * global_batch), ("validation", heldout_blocks), ("test", heldout_blocks)):
        order, quota = mixture_order(count, 42 + {"train": 0, "validation": 1, "test": 2}[split])
        np.save(root / f"{split}-order.npy", order, allow_pickle=False)
        quotas[split] = quota
        split_metadata[split] = dict(blocks=count, order_sha256=sha256(root / f"{split}-order.npy"))
    manifest = dict(recipe=RECIPE_URL, sequence_length=args.sequence_length, steps=args.steps,
        global_batch_sequences=global_batch, data_seed=42, tokenizer="HuggingFaceTB/cosmo2-tokenizer",
        tokenizer_revision=args.tokenizer_revision, tokenizer_sha256=sha256(tokenizer_path),
        format="little-endian uint32", weights=WEIGHTS, splits=split_metadata,
        packing=f"EOS-separated documents; {args.sequence_length} inputs and shifted targets; no padding; attention mask specified by trainer",
        deviations=["Public BigCode StackOverflow substitute: original snapshot unspecified",
                    "Public Avelina Python-Edu substitute: smollm-corpus python-edu stores only blob IDs without text",
                    "Reduced shuffled public shard subset, not original full-corpus token order",
                    "SHA256 document split: 98% train / 1% validation / 1% test"], sources={})
    api = HfApi()
    for source_index, (name, (repo, revision, prefix)) in enumerate(SOURCES.items()):
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                       if f.startswith(prefix) and f.endswith(".parquet"))
        if not files:
            raise ValueError(f"no parquet shards: {repo}/{prefix}")
        np.random.RandomState(42).shuffle(files)
        target = {split: int(quota[source_index]) * args.sequence_length + 1 for split, quota in quotas.items()}
        paths = {split: root / f"{name}-{split}.bin" for split in quotas}
        counts = dict.fromkeys(quotas, 0)
        documents = dict.fromkeys(quotas, 0)
        handles = {split: path.open("wb") for split, path in paths.items()}
        used_files = []
        try:
            for filename in files:
                print(f"Downloading {name}: {filename}", flush=True)
                path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
                used_files.append(dict(file=filename, sha256=sha256(path)))
                parquet = pq.ParquetFile(path)
                fields = parquet.schema_arrow.names
                column = "text" if "text" in fields else "content" if "content" in fields else None
                if column is None:
                    raise ValueError(f"no supported text column: {fields}")
                for batch in parquet.iter_batches(batch_size=256, columns=[column]):
                    texts = [t for t in batch.column(0).to_pylist() if isinstance(t, str) and t.strip()]
                    for text, encoded in zip(texts, tokenizer.encode_batch(texts, add_special_tokens=False)):
                        split = document_split(text)
                        if counts[split] >= target[split]:
                            continue
                        ids = encoded.ids + [0]
                        if max(ids) >= 49152:
                            raise ValueError("out-of-vocabulary token")
                        ids = ids[:target[split] - counts[split]]
                        np.asarray(ids, dtype="<u4").tofile(handles[split])
                        counts[split] += len(ids)
                        documents[split] += 1
                    if all(counts[s] >= target[s] for s in quotas):
                        break
                print(json.dumps(dict(source=name, tokens=counts, required=target)), flush=True)
                if all(counts[s] >= target[s] for s in quotas):
                    break
        finally:
            for handle in handles.values():
                handle.close()
        if any(counts[s] < target[s] for s in quotas):
            raise ValueError(f"insufficient source data: {name}")
        manifest["sources"][name] = dict(repo=repo, revision=revision, files=used_files,
            splits={s: dict(file=paths[s].name, tokens=counts[s], documents=documents[s], sha256=sha256(paths[s])) for s in quotas})
        (root / "progress.json").write_text(json.dumps(manifest, indent=2))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for split in quotas:
        data = MixedBlocks(root, split)
        assert data[0].shape == (args.sequence_length + 1,)
    print("PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
