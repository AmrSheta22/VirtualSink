"""Document-preserving token preparation for recurrent-memory training.

The existing packed-block corpus has no authoritative document offsets. This
format keeps explicit boundaries rather than inferring them from EOS contents.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from .data import sha256


class TokenDocuments:
    def __init__(self, directory, split, vocab_size):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "document-uint32-v1":
            raise ValueError("recurrent training requires explicit document boundaries")
        entry = self.manifest["splits"][split]
        token_path = directory / entry["file"]
        offset_path = directory / entry["offsets_file"]
        if sha256(token_path) != entry["sha256"] or sha256(offset_path) != entry["offsets_sha256"]:
            raise ValueError(f"{split} corpus checksum mismatch")
        if token_path.stat().st_size % 4:
            raise ValueError("truncated uint32 token stream")
        self.tokens = np.memmap(token_path, dtype="<u4", mode="r")
        self.offsets = np.load(offset_path, mmap_mode="r", allow_pickle=False)
        if (self.offsets.dtype != np.dtype("int64") or self.offsets.ndim != 1
                or len(self.offsets) < 2 or self.offsets[0] != 0
                or self.offsets[-1] != self.tokens.size or np.any(np.diff(self.offsets) < 2)
                or entry["tokens"] != self.tokens.size or entry["documents"] != len(self)):
            raise ValueError("invalid document offsets/counts")
        eos = self.manifest["eos_token_id"]
        if not 0 <= eos < vocab_size or np.any(self.tokens[self.offsets[1:] - 1] != eos):
            raise ValueError("document must end in the declared EOS token")
        for start in range(0, self.tokens.size, 1_000_000):
            if self.tokens[start:start + 1_000_000].max() >= vocab_size:
                raise ValueError("token ID outside model vocabulary")

    def __len__(self):
        return len(self.offsets) - 1

    def length(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return int(self.offsets[index + 1] - self.offsets[index])

    def segment(self, index, start, count):
        """Return adjacent input/target sequences without crossing a document."""
        length = self.length(index)
        if not 0 <= start < length - 1 or count < 1:
            raise ValueError("invalid segment start/count")
        left = int(self.offsets[index]) + start
        right = min(left + count + 1, int(self.offsets[index + 1]))
        tokens = torch.from_numpy(self.tokens[left:right].astype(np.int64))
        return tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0)


def prepare_split(source, directory, split, tokenizer):
    token_path, offset_path = directory / f"{split}.bin", directory / f"{split}.offsets.npy"
    offsets = [0]
    with source.open(encoding="utf-8") as records, token_path.open("wb") as output:
        for line in records:
            text = json.loads(line)["text"]
            if not isinstance(text, str):
                raise ValueError("JSONL text must be a string")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(tokens) < 2:
                continue
            np.asarray(tokens, dtype="<u4").tofile(output)
            offsets.append(offsets[-1] + len(tokens))
    if len(offsets) < 2:
        raise ValueError(f"{split} has no documents")
    np.save(offset_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    return {"file": token_path.name, "offsets_file": offset_path.name,
            "tokens": offsets[-1], "documents": len(offsets) - 1,
            "sha256": sha256(token_path), "offsets_sha256": sha256(offset_path),
            "source_sha256": sha256(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source", required=True, help="dataset name/revision/license and split provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable model commit SHA")
    paths = [getattr(args, split).resolve() for split in ("train", "validation", "test")]
    if len({sha256(path) for path in paths}) != 3:
        parser.error("source splits must be distinct")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.eos_token_id is None:
        parser.error("tokenizer must define EOS")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "document-uint32-v1", "model": args.model, "revision": args.revision,
                "source": args.source, "eos_token_id": tokenizer.eos_token_id,
                "packing": "explicit document offsets; reset KV and recurrent memory at each boundary",
                "splits": {}}
    for split, path in zip(("train", "validation", "test"), paths):
        manifest["splits"][split] = prepare_split(path, args.output_dir, split, tokenizer)
    if len({entry["sha256"] for entry in manifest["splits"].values()}) != 3:
        raise ValueError("tokenized splits must be distinct")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
