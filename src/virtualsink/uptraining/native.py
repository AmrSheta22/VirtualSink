"""FP32 Llama/Qwen2 decoder for the PyTorch 1.12 K80 environment.

Loads original safetensors with original parameter names. Supported configs are
deliberately narrow; numerical parity with Transformers is tested on CPU.
No Transformers dependency is imported on the legacy training runtime.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .qwen import sink_attention


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon) * self.weight


class Attention(nn.Module):
    def __init__(self, config, chunk_size):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.heads)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.chunk_size = chunk_size
        bias = config.model_type == "qwen2" or getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size,
                                bias=getattr(config, "attention_bias", False))
        self.sink_w = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.sink_b = nn.Parameter(torch.full((self.heads,), -3.5))
        nn.init.normal_(self.sink_w, std=1e-4)
        self.sink_disabled = False
        self.window_size = None

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = [p(x).view(batch, length, -1, self.head_dim).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj)]
        inv = 1 / self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim)
        phase = torch.arange(length, device=x.device).float()[:, None] * inv[None, :]
        phase = torch.cat((phase, phase), -1)[None, None]
        def rotary(t):
            half = self.head_dim // 2
            rotated = torch.cat((-t[..., half:], t[..., :half]), -1)
            return t * phase.cos() + rotated * phase.sin()
        q, k = rotary(q), rotary(k)
        groups = self.heads // self.kv_heads
        k, v = [t.repeat_interleave(groups, dim=1) for t in (k, v)]
        w, b = self.sink_w, self.sink_b
        if self.sink_disabled:
            w, b = torch.zeros_like(w), torch.full_like(b, -1e6)
        out, gates = sink_attention(q, k, v, w, b, self.chunk_size,
                                    "reentrant" if self.training and self.window_size is None else False, self.window_size)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1)), gates


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config, chunk_size):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, chunk_size)
        self.mlp = MLP(config)

    def forward(self, x):
        update, gate = self.self_attn(self.input_layernorm(x))
        x = x + update
        return x + self.mlp(self.post_attention_layernorm(x)), (gate.detach() if self.training else gate)


class Backbone(nn.Module):
    def __init__(self, config, chunk_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Layer(config, chunk_size) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.checkpointing = False

    def forward(self, input_ids, use_cache=False, output_attentions=True, return_dict=True):
        if use_cache:
            raise ValueError("KV cache not supported")
        x, gates = self.embed_tokens(input_ids), []
        for layer in self.layers:
            if self.training and self.checkpointing:
                if not x.requires_grad:
                    x.requires_grad_(True)
                x, gate = checkpoint(layer, x, use_reentrant=True)
            else:
                x, gate = layer(x)
            gates.append(gate)
        return SimpleNamespace(last_hidden_state=self.norm(x), attentions=tuple(gates))


class NativeLM(nn.Module):
    def __init__(self, config, chunk_size=64):
        super().__init__()
        self.config = SimpleNamespace(**config)
        c = self.config
        if c.model_type not in ("llama", "qwen2") or c.hidden_act != "silu":
            raise ValueError("unsupported model architecture")
        if (getattr(c, "rope_scaling", None) or getattr(c, "use_sliding_window", False)
                or getattr(c, "attention_dropout", 0) or getattr(c, "pretraining_tp", 1) != 1):
            raise ValueError("unsupported RoPE, window, dropout or tensor-parallel config")
        self.model = Backbone(c, chunk_size)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.checkpointing = True


def load_native(directory, chunk_size=64):
    from safetensors.torch import load_file
    directory = Path(directory)
    model = NativeLM(json.loads((directory / "config.json").read_text()), chunk_size)
    state = {}
    index = directory / "model.safetensors.index.json"
    files = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
             if index.exists() else ["model.safetensors"])
    for name in files:
        state.update(load_file(str(directory / name), device="cpu"))
    if model.config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.endswith((".sink_w", ".sink_b")) for k in result.missing_keys):
        raise ValueError(f"pretrained checkpoint mismatch: {result}")
    return model


def set_window(model, window_size=None):
    if window_size is not None and (isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0):
        raise ValueError("window_size must be positive or None")
    for layer in model.model.layers:
        layer.self_attn.window_size = window_size
