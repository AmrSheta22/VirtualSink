"""Deterministic, on-demand trigger-conditional regression sequences."""

from typing import Dict

import torch
from torch.utils.data import Dataset


class TriggerConditionalDataset(Dataset):
    """One trigger per sequence, with zero targets everywhere else, including BOS.

    Each index has its own RNG seed, so access order and DataLoader workers do
    not change examples. Use distinct seeds for training, validation, and test.
    At the trigger the target averages the complete input vectors in [1, j),
    including their context indicator, exactly as specified by the task.
    Gaussian features at BOS and at the trigger are retained.
    """

    def __init__(self, num_samples: int = 192000, d: int = 64,
                 seq_len: int = 64, seed: int = 0) -> None:
        for name, value, minimum in (("num_samples", num_samples, 1),
                                     ("d", d, 4), ("seq_len", seq_len, 4)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.num_samples, self.d, self.seq_len, self.seed = num_samples, d, seq_len, seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Disjoint seed strides give independent streams without storing tensors.
        generator = torch.Generator().manual_seed((self.seed + index * 1000003) % (2**63 - 1))
        j = int(torch.randint(2, self.seq_len - 1, (), generator=generator))
        x = torch.zeros(self.seq_len, self.d)
        x[:, 3:] = torch.randn(self.seq_len, self.d - 3, generator=generator)
        x[0, 0] = 1
        x[j, 1] = 1
        x[:, 2] = 1
        x[[0, j], 2] = 0
        y = torch.zeros_like(x)
        y[j] = x[1:j].mean(dim=0)
        trigger = torch.zeros(self.seq_len, dtype=torch.bool)
        trigger[j] = True
        return {"x": x, "y": y, "trigger_mask": trigger,
                "trigger_index": torch.tensor(j, dtype=torch.long)}
