"""Token and coordinate aware regression metrics and objectives."""

from typing import Dict

import torch


class MSEAccumulator:
    """Accumulate element-weighted MSE; never average unequal batch means."""

    def __init__(self) -> None:
        self.sums = {prefix + name: 0.0 for prefix in ("", "gaussian_", "indicator_")
                     for name in ("mse", "trigger_mse", "non_trigger_mse")}
        self.counts = dict.fromkeys(self.sums, 0)

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               trigger_mask: torch.Tensor) -> None:
        error = (prediction.detach().float() - target.float()).square()
        assert torch.isfinite(error).all(), "Non-finite prediction or squared error"
        if error.shape[-1] < 4:
            raise ValueError("metrics require indicator coordinates 0:3 and Gaussian features 3:")
        for prefix, coordinates in (("", error), ("gaussian_", error[..., 3:]),
                                    ("indicator_", error[..., :3])):
            per_token = coordinates.mean(dim=-1)
            for name, values in (("mse", per_token), ("trigger_mse", per_token[trigger_mask]),
                                 ("non_trigger_mse", per_token[~trigger_mask])):
                key = prefix + name
                self.sums[key] += values.double().sum().item()
                self.counts[key] += values.numel()

    def compute(self) -> Dict[str, float]:
        if not all(self.counts.values()):
            raise ValueError("metrics require both trigger and non-trigger positions")
        result = {name: self.sums[name] / self.counts[name] for name in self.sums}
        for prefix in ("", "gaussian_", "indicator_"):
            result[prefix + "balanced_mse"] = 0.5 * (
                result[prefix + "trigger_mse"] + result[prefix + "non_trigger_mse"])
        return result


def regression_loss(prediction: torch.Tensor, target: torch.Tensor,
                    trigger_mask: torch.Tensor, mode: str = "token_mean") -> torch.Tensor:
    """Both losses use all coordinates; only trigger/non-trigger weighting changes."""
    per_token = (prediction - target).square().mean(dim=-1)
    if mode == "token_mean":
        return per_token.mean()
    if mode == "balanced":
        return 0.5 * (per_token[trigger_mask].mean() + per_token[~trigger_mask].mean())
    raise ValueError(f"unknown loss mode: {mode}")
