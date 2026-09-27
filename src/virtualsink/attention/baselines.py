"""Matched softmax, fixed sink, and learned scalar sink baselines."""

import torch

from .query_sink import QueryVirtualSinkAttention


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
