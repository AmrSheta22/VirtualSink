"""Causal attention with query-conditioned, zero-value virtual sinks."""

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn


class QueryVirtualSinkAttention(nn.Module):
    """Multi-head attention whose physical attention mass may be less than one.

    Inputs are unprojected (B, T, d_model) tensors. Separate q/k/v inputs
    support differing query/key lengths: queries align with the *last* Tq
    positions in the key sequence, as in decoding with a prefix. Tk >= Tq.
    No KV cache or virtual key/value tensor is allocated by this module.

    Boolean attention masks use True for allowed entries; floating masks are
    additive logits (use -inf to exclude a key). Masks must broadcast to
    (B, H, Tq, Tk); a batch padding mask should be shaped (B, 1, 1, Tk).
    Masks affect only physical keys, never the sink. Fully masked rows are safe.
    Projections have no bias so a fully dominant sink yields exactly zero.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: Optional[int] = None,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        sink_bias_init: float = -3.5,
    ) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads)):
            self._positive_integer(name, value)
        if d_head is None:
            if d_model % n_heads:
                raise ValueError("d_model must be divisible by n_heads without d_head")
            d_head = d_model // n_heads
        self._positive_integer("d_head", d_head)
        if window_size is not None:
            self._positive_integer("window_size", window_size)
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if not math.isfinite(sink_bias_init):
            raise ValueError("sink_bias_init must be finite")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        inner = n_heads * d_head
        self.q_proj = nn.Linear(d_model, inner, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, inner, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, inner, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(inner, d_model, bias=False, dtype=dtype)
        self.sink_w = nn.Parameter(torch.empty(n_heads, d_head, dtype=dtype))
        self.sink_b = nn.Parameter(torch.full((n_heads,), sink_bias_init, dtype=dtype))
        nn.init.normal_(self.sink_w, mean=0.0, std=1e-4)

    @staticmethod
    def _positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        *,
        q: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Accept ``module(x)``, ``module(q, k, v)``, or named q/k/v.

        A non-None sliding_window overrides the constructor window. This is
        a dense reference implementation: local masking does not reduce its
        quadratic score memory. Low-precision logits are computed in FP32.
        return_attention additionally returns detached physical weights, sink
        probability, sink logits, and physical log-partition for diagnostics.
        """
        if q is not None:
            if x is not None:
                raise ValueError("provide x or q, not both")
            x = q
        if x is None:
            raise ValueError("x or q is required")
        if (k is None) != (v is None):
            raise ValueError("k and v must be provided together")
        if k is None:
            k = v = x
        assert v is not None
        for tensor in (x, k, v):
            if tensor.ndim != 3 or tensor.shape[-1] != self.d_model:
                raise ValueError("inputs must have shape (B, T, d_model)")
            if tensor.device != x.device or tensor.dtype != x.dtype:
                raise ValueError("inputs must share device and dtype")
        if k.shape != v.shape or k.shape[0] != x.shape[0]:
            raise ValueError("k/v shapes and input batch sizes must match")
        batch, tq, _ = x.shape
        tk = k.shape[1]
        if tq == 0 or tk < tq:
            raise ValueError("sequence lengths must satisfy Tk >= Tq > 0")
        window = self.window_size if sliding_window is None else sliding_window
        if window is not None:
            self._positive_integer("sliding_window", window)

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, -1, self.n_heads, self.d_head).transpose(1, 2)

        query, key, value = split(self.q_proj(x)), split(self.k_proj(k)), split(self.v_proj(v))
        # Disabling autocast here also protects logits inside an outer AMP context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
            query, key = query.to(compute_dtype), key.to(compute_dtype)
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_head)
            sink = self._sink_logits(query)
            qi = torch.arange(tq, device=x.device)[:, None] + tk - tq
            ki = torch.arange(tk, device=x.device)[None, :]
            allowed = ki <= qi
            if window is not None:
                allowed = allowed & (ki >= qi - window + 1)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attention_mask, -torch.inf)
                elif attention_mask.is_floating_point():
                    scores = scores + attention_mask.to(compute_dtype)
                else:
                    raise ValueError("attention_mask must be boolean or floating point")
                if scores.shape != (batch, self.n_heads, tq, tk):
                    raise ValueError("attention_mask must broadcast to (B, H, Tq, Tk)")
            scores = scores.masked_fill(~allowed, -torch.inf)
            logits = scores if sink is None else torch.cat((scores, sink.unsqueeze(-1)), dim=-1)
            weights = torch.softmax(logits, dim=-1)
            physical_weights = weights if sink is None else weights[..., :-1]
            output = physical_weights @ value.to(compute_dtype)
        output = output.transpose(1, 2).reshape(batch, tq, -1).to(value.dtype)
        output = self.out_proj(output)
        if return_attention:
            diagnostics = {
                "physical_weights": physical_weights.detach(),
                "sink_probability": (torch.zeros_like(weights[..., 0]) if sink is None
                                     else weights[..., -1]).detach(),
                "physical_log_partition": torch.logsumexp(scores, dim=-1).detach(),
            }
            if sink is not None:
                diagnostics["sink_logits"] = sink.detach()
            return output, diagnostics
        return output

    def _sink_logits(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the query-conditioned sink logit; baselines override this hook."""
        return (torch.einsum("bhtd,hd->bht", query, self.sink_w.to(query.dtype))
                + self.sink_b.to(query.dtype).view(1, -1, 1))
