"""Matched toy Transformers differing only in their attention denominator."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .attention import ATTENTION_TYPES


class AttentionOnlyModel(nn.Module):
    """One attention layer; no residual, normalization, MLP, or extra head.

    Q/K/V/output projections remain trainable and matched across attention arms.
    This removes alternative token-wise nonlinear paths without artificially
    forcing BOS attention or prescribing sink probabilities.
    """

    def __init__(self, attn_type: str = "query_virtual_sink", d_model: int = 64,
                 n_heads: int = 4, d_head: Optional[int] = None,
                 num_layers: int = 1) -> None:
        super().__init__()
        if attn_type not in ATTENTION_TYPES:
            raise ValueError(f"unknown attention type: {attn_type}")
        if num_layers != 1:
            raise ValueError("attention_only requires num_layers=1")
        self.config = dict(architecture="attention_only", attn_type=attn_type,
                           d_model=d_model, n_heads=n_heads, d_head=d_head, num_layers=1)
        self.attn = ATTENTION_TYPES[attn_type](d_model, n_heads, d_head=d_head)

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        if return_attention:
            prediction, info = self.attn(x, return_attention=True)
            return prediction, [info]
        return self.attn(x)


def build_model(config: dict) -> nn.Module:
    """Load either architecture; old Transformer checkpoint configs still work."""
    options = dict(config)
    architecture = options.pop("architecture", "transformer")
    if architecture == "attention_only":
        return AttentionOnlyModel(**options)
    if architecture == "transformer":
        return ToyTransformer(**options)
    raise ValueError(f"unknown architecture: {architecture}")


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention plus a residual position-wise MLP."""

    def __init__(self, d_model: int, n_heads: int, d_head: Optional[int],
                 attn_type: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ATTENTION_TYPES[attn_type](d_model, n_heads, d_head=d_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        result = self.attn(self.norm1(x), return_attention=return_attention)
        if return_attention:
            update, diagnostics = result
        else:
            update = result
        x = x + update
        x = x + self.mlp(self.norm2(x))
        return (x, diagnostics) if return_attention else x


class ToyTransformer(nn.Module):
    """Small regression Transformer with the same architecture for every arm.

    No positional embeddings are necessary for this permutation-invariant prefix
    mean task; the causal mask supplies prefix boundaries. Residual/MLP paths can
    solve no-op behavior independently, so attention patterns are measurements,
    not architectural or theoretical guarantees.
    """

    def __init__(self, attn_type: str = "query_virtual_sink", d_model: int = 64,
                 n_heads: int = 4, d_head: Optional[int] = None,
                 num_layers: int = 2) -> None:
        super().__init__()
        if attn_type not in ATTENTION_TYPES:
            raise ValueError(f"attn_type must be one of {tuple(ATTENTION_TYPES)}")
        if num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")
        self.config = dict(attn_type=attn_type, d_model=d_model, n_heads=n_heads,
                           d_head=d_head, num_layers=num_layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_head, attn_type) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        diagnostics = []
        for block in self.blocks:
            if return_attention:
                x, info = block(x, return_attention=True)
                diagnostics.append(info)
            else:
                x = block(x)
        prediction = self.output_head(x)
        return (prediction, diagnostics) if return_attention else prediction
