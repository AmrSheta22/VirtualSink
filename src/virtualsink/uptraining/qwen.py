"""Training-only Qwen2.5 adapter for Transformers 4.51.3.

Attention diagnostics are physical gates (B,H,T), NOT attention matrices.
No KV cache or padding is supported by this packed-sequence reference backend.
"""

import types

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def sink_attention(q, k, v, w, b, chunk_size=128, recompute=True):
    """Exact zero-value sink softmax, with query-chunk activation recomputation.

    GQA keys/values must already be repeated. FP32 normalization under AMP.
    Chunking reduces live score storage, not quadratic arithmetic.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    length = q.shape[-2]
    if k.shape != v.shape or q.shape != k.shape or length == 0:
        raise ValueError("expected matching nonempty (B,H,T,D) Q/K/V")

    def attend(qc, key, value, weight, bias, positions):
        with torch.autocast(q.device.type, enabled=False):
            dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
            qc, key = qc.to(dtype), key.to(dtype)
            scores = (qc @ key.transpose(-1, -2)) * q.shape[-1] ** -0.5
            mask = torch.arange(length, device=q.device)[None, :] > positions[:, None]
            scores = scores.masked_fill(mask, -torch.inf)
            sink = (qc * weight.to(dtype)[None, :, None]).sum(-1) + bias.to(dtype)[None, :, None]
            lse = torch.logsumexp(scores, -1)
            gate = torch.sigmoid(lse - sink)
            weights = torch.softmax(torch.cat((scores, sink[..., None]), -1), -1)[..., :-1]
            output = weights @ value.to(dtype)
        return output.to(value.dtype), gate

    outputs, gates = [], []
    for start in range(0, length, chunk_size):
        args = (q[:, :, start:start + chunk_size], k, v, w, b,
                torch.arange(start, min(start + chunk_size, length), device=q.device))
        out, gate = (checkpoint(attend, *args, use_reentrant=False)
                     if recompute and torch.is_grad_enabled() else attend(*args))
        outputs.append(out)
        gates.append(gate)
    return torch.cat(outputs, -2), torch.cat(gates, -1)


def _forward(self, hidden_states, position_embeddings, attention_mask=None,
             past_key_value=None, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    if past_key_value is not None or kwargs.get("use_cache", False):
        raise ValueError("Phase 3 reference adapter requires use_cache=False")
    # HF's causal mask is redundant here. Reject padding/custom masks instead
    # of silently changing their meaning; the runner sends packed full blocks.
    if attention_mask is not None:
        n = hidden_states.shape[1]
        if attention_mask.ndim != 4 or attention_mask.shape[-2] != n or attention_mask.shape[-1] < n:
            raise ValueError("only full, unpadded causal sequences are supported")
        attention_mask = attention_mask[..., :n]
        allowed = torch.ones(n, n, dtype=torch.bool, device=hidden_states.device).tril()
        if not ((attention_mask[..., allowed] == 0).all()
                and (attention_mask[..., ~allowed] < -1e4).all()):
            raise ValueError("padding or custom masks are not supported")
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    q, k, v = [proj(hidden_states).view(shape).transpose(1, 2)
               for proj in (self.q_proj, self.k_proj, self.v_proj)]
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k, v = [repeat_kv(x, self.num_key_value_groups) for x in (k, v)]
    w, b = self.sink_w, self.sink_b
    if getattr(self, "sink_disabled", False):
        w, b = torch.zeros_like(w), torch.full_like(b, -1e6)
    out, gates = sink_attention(q, k, v, w, b,
                                self.sink_chunk_size, self.training)
    out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    return self.o_proj(out), gates


def adapt_qwen(model, chunk_size=128, sink_bias=-3.5):
    """Keep every pretrained tensor intact; insert sink parameters after RoPE."""
    import transformers

    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Phase 3 requires transformers==4.51.3; adapter API is pinned")
    if model.config.model_type != "qwen2" or model.config.use_sliding_window:
        raise ValueError("only full-attention Qwen2/Qwen2.5 is supported")
    if model.config.attention_dropout != 0:
        raise ValueError("reference backend requires attention_dropout=0")
    if chunk_size <= 0 or not torch.isfinite(torch.tensor(sink_bias)):
        raise ValueError("invalid chunk size or sink bias")
    if any(hasattr(layer.self_attn, "sink_w") for layer in model.model.layers):
        raise ValueError("model is already adapted")
    for layer in model.model.layers:
        attn = layer.self_attn
        device = attn.q_proj.weight.device
        # Keep optimizer-owned sink weights in FP32 even with BF16 autocast.
        attn.sink_w = nn.Parameter(torch.empty(model.config.num_attention_heads,
                                              attn.head_dim, device=device))
        attn.sink_b = nn.Parameter(torch.full((model.config.num_attention_heads,),
                                             float(sink_bias), device=device))
        nn.init.normal_(attn.sink_w, std=1e-4)
        attn.sink_chunk_size = chunk_size
        attn.forward = types.MethodType(_forward, attn)
    model.config.use_cache = False
    model.config._attn_implementation = "eager"
    return model


def disable_sinks(model, disabled=True):
    """Standard-attention control through the same numerical backend."""
    for layer in model.model.layers:
        layer.self_attn.sink_disabled = disabled


def parameter_groups(model, mode="attention", base_lr=2e-5, sink_lr=1e-4):
    if mode not in ("attention", "full"):
        raise ValueError("mode must be attention or full")
    base, sinks = [], []
    for name, param in model.named_parameters():
        sink = name.endswith((".sink_w", ".sink_b"))
        projection = any(f".self_attn.{p}." in name for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
        param.requires_grad_(sink or mode == "full" or projection)
        if param.requires_grad:
            (sinks if sink else base).append(param)
    if not sinks:
        raise ValueError("adapt model before configuring training")
    return [{"params": base, "lr": base_lr, "weight_decay": 0.01},
            {"params": sinks, "lr": sink_lr, "weight_decay": 0.0}]


def auxiliary_loss(gates, kind="anti_collapse", epsilon=1e-6):
    """Average layer penalties; each head mean is over batch and query tokens.

    Anti-collapse discourages low mean physical mass; it does not equalize heads.
    The supplied objective is retained only as an explicitly selected ablation.
    """
    means = torch.stack([g.float().mean(dim=(0, 2)) for g in gates])
    if kind == "anti_collapse":
        loss = -torch.log(means + epsilon).mean()
    elif kind == "supplied":
        loss = (means * -torch.log(means + epsilon)).sum(-1).mean() * means.shape[-1]
    elif kind == "none":
        loss = means.sum() * 0
    else:
        raise ValueError("unknown auxiliary loss")
    return loss, means
