"""Matched toy Transformers differing only in their attention denominator."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from query_virtual_sink_attention import QueryVirtualSinkAttention


class StandardSoftmaxAttention(QueryVirtualSinkAttention):
    """Ordinary causal softmax. BOS is available but its use is not forced."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.sink_w
        del self.sink_b

    def _sink_logits(self, query: torch.Tensor) -> None:
        return None


class QuietSoftmaxAttention(StandardSoftmaxAttention):
    """Softmax1: an exact constant +1 in the denominator, with zero value."""

    def _sink_logits(self, query: torch.Tensor) -> torch.Tensor:
        return query.new_zeros(query.shape[:-1])


class StaticSinkAttention(QueryVirtualSinkAttention):
    """Learnable per-head scalar sink, independent of each query."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.sink_w

    def _sink_logits(self, query: torch.Tensor) -> torch.Tensor:
        return self.sink_b.to(query.dtype).view(1, -1, 1).expand(query.shape[:-1])


ATTENTION_TYPES = {
    "standard": StandardSoftmaxAttention,
    "quiet": QuietSoftmaxAttention,
    "static_sink": StaticSinkAttention,
    "query_virtual_sink": QueryVirtualSinkAttention,
}


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
