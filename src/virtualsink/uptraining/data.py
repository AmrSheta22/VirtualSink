"""Auditable local JSONL -> packed uint32 token streams, with bounded memory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenBlocks(Dataset):
    def __init__(self, path, seq_len, vocab_size):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.path = Path(path)
        if self.path.stat().st_size % 4:
            raise ValueError("token file is not a uint32 stream")
        self.tokens = np.memmap(self.path, dtype="<u4", mode="r")
        self.seq_len = seq_len
        if len(self) == 0:
            raise ValueError("not enough tokens for one block")
        for start in range(0, self.tokens.size, 1_000_000):
            if self.tokens[start:start + 1_000_000].max() >= vocab_size:
                raise ValueError("token ID outside model vocabulary")

    def __len__(self):
        return self.tokens.size // self.seq_len

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_len
        return torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True, help="immutable model commit SHA")
    parser.add_argument("--source", required=True, help="dataset name, revision, license and split policy")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable 40-character commit SHA")
    paths = [args.train.resolve(), args.validation.resolve(), args.test.resolve()]
    hashes = [sha256(p) for p in paths]
    if len(set(hashes)) != 3:
        parser.error("splits must be distinct; duplicate source files detected")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires EOS")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "little-endian uint32", "model": args.model,
                "revision": args.revision, "source": args.source,
                "packing": "EOS-separated documents; cross-document attention allowed; no padding",
                "splits": {}}
    for split, source, digest in zip(("train", "validation", "test"), paths, hashes):
        destination = args.output_dir / f"{split}.bin"
        count, documents = 0, 0
        with source.open(encoding="utf-8") as inp, destination.open("wb") as out:
            for line in inp:
                record = json.loads(line)
                text = record["text"]
                if not isinstance(text, str):
                    raise ValueError("JSONL text must be a string")
                if not text.strip():
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                np.asarray(ids, dtype="<u4").tofile(out)
                count += len(ids)
                documents += 1
        if count == 0:
            raise ValueError(f"empty {split} split")
        manifest["splits"][split] = {"file": destination.name, "tokens": count,
                                      "documents": documents, "sha256": sha256(destination),
                                      "source_sha256": digest}
        print(f"{split}: {count} tokens from {documents} documents", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
