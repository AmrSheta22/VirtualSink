"""Portable model state loading, including legacy project checkpoints."""

from pathlib import Path

import torch


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> dict:
    # The compatibility fallback is for this project's own checkpoints on torch 1.12.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
