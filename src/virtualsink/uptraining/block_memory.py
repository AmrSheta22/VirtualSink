"""Causal eviction-block summaries: mean, received-attention, and learned pooling.

FP32 reference, aligned unpadded streams, absolute physical RoPE. Summaries use
unrotated keys and no positional encoding. Every completed eviction block adds
three K/V slots; one shared affine null sink has identically zero value.
"""
from dataclasses import dataclass, fields
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .native import NativeLM
from .recurrent_sink import _rotate


@dataclass(frozen=True)
class BlockLayerState:
    key: torch.Tensor
    value: torch.Tensor
    importance: torch.Tensor
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    sum_key: torch.Tensor
    sum_value: torch.Tensor
    weighted_key: torch.Tensor
    weighted_value: torch.Tensor
    weight_sum: torch.Tensor
    free_key: torch.Tensor
    free_value: torch.Tensor
    free_sum: torch.Tensor
    free_max: torch.Tensor

    def tensors(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def detach(self):
        return BlockLayerState(*(x.detach() for x in self.tensors()))


@dataclass(frozen=True)
class BlockState:
    layers: tuple
    position: int
    capacity: int
    block_size: int

    def detach(self):
        return BlockState(tuple(s.detach() for s in self.layers), self.position,
                          self.capacity, self.block_size)


@dataclass(frozen=True)
class BlockOutput:
    hidden: torch.Tensor
    state: BlockState
    diagnostics: object = None


class BlockBank(nn.Module):
    def __init__(self, heads, width):
        super().__init__()
        self.width = width
        self.pool_query = nn.Parameter(torch.empty(heads, width))
        nn.init.normal_(self.pool_query, std=width ** -.5)
        self.role_bias = nn.Parameter(torch.full((heads, 3), -8.))
        self.null_weight = nn.Parameter(torch.zeros(heads, width))
        self.null_bias = nn.Parameter(torch.full((heads,), -8.))

    def accumulate(self, saved, key, value, importance, count, block_size):
        # epsilon makes zero-importance blocks reduce to the uniform mean.
        weight = importance.unsqueeze(-1) + 1e-8
        sk, sv = saved.sum_key + key, saved.sum_value + value
        wk, wv = saved.weighted_key + weight * key, saved.weighted_value + weight * value
        ws = saved.weight_sum + weight
        score = (key * self.pool_query).sum(-1, keepdim=True) * self.width ** -.5
        maximum = torch.maximum(saved.free_max, score)
        old_scale, new_scale = (saved.free_max - maximum).exp(), (score - maximum).exp()
        fk = old_scale * saved.free_key + new_scale * key
        fv = old_scale * saved.free_value + new_scale * value
        fs = old_scale * saved.free_sum + new_scale
        mk, mv = saved.memory_key, saved.memory_value
        if count == block_size:
            mk = torch.cat((mk, torch.stack((sk / count, wk / ws, fk / fs), 2)), 2)
            mv = torch.cat((mv, torch.stack((sv / count, wv / ws, fv / fs), 2)), 2)
            sk, sv, wk, wv, ws, fk, fv, fs = (torch.zeros_like(x) for x in (sk, sv, wk, wv, ws, fk, fv, fs))
            maximum = torch.full_like(maximum, -torch.inf)
        return BlockLayerState(saved.key, saved.value, saved.importance, mk, mv,
                               sk, sv, wk, wv, ws, fk, fv, fs, maximum)


class BlockMemoryLM(nn.Module):
    """Reference implementation; memory bank grows by three slots per block.

    Importance is attention received over a token's physical-cache lifetime,
    including its own causal query. Writes publish before the current read.
    Partially evicted blocks remain unreadable until their last token exits.
    """
    def __init__(self, base, capacity=128, block_size=None):
        super().__init__()
        block_size = capacity if block_size is None else block_size
        if not isinstance(base, NativeLM):
            raise TypeError("requires NativeLM")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (capacity, block_size)):
            raise ValueError("capacity and block_size must be positive integers")
        if any(layer.self_attn.head_dim % 2 for layer in base.model.layers):
            raise ValueError("requires even RoPE width")
        self.base, self.capacity, self.block_size = base, capacity, block_size
        self.memory = nn.ModuleList([BlockBank(l.self_attn.heads, l.self_attn.head_dim)
                                     for l in base.model.layers])
        self.memory.to(base.model.embed_tokens.weight.device)
        self.checkpointing = False

    def get_extra_state(self):
        return dict(version=1, capacity=self.capacity, block_size=self.block_size,
                    architecture="eviction_block_mean_importance_learned_v1")

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("block-memory checkpoint architecture mismatch")

    def parameter_groups(self, mode="memory", base_lr=2e-5, memory_lr=1e-3):
        if mode not in ("memory", "attention", "full"):
            raise ValueError("invalid parameter mode")
        if any(not math.isfinite(v) or v <= 0 for v in (base_lr, memory_lr)):
            raise ValueError("learning rates must be finite and positive")
        selected = []
        for name, p in self.base.named_parameters():
            unused = name.endswith((".sink_w", ".sink_b", ".sink_v"))
            projection = any(f".self_attn.{n}." in name for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
            p.requires_grad_(not unused and (mode == "full" or (mode == "attention" and projection)))
            if p.requires_grad:
                selected.append(p)
        for p in self.memory.parameters():
            p.requires_grad_(True)
        groups = [dict(params=list(self.memory.parameters()), lr=memory_lr, weight_decay=0.)]
        if selected:
            groups.append(dict(params=selected, lr=base_lr, weight_decay=.01))
        return groups

    def initial_state(self, batch):
        device = self.base.model.embed_tokens.weight.device
        states = []
        for layer in self.base.model.layers:
            a = layer.self_attn
            def zeros(*shape):
                return torch.zeros(*shape, device=device)
            kv = zeros(batch, a.kv_heads, 0, a.head_dim)
            mem = zeros(batch, a.heads, 0, a.head_dim)
            vec = zeros(batch, a.heads, a.head_dim)
            scalar = zeros(batch, a.heads, 1)
            states.append(BlockLayerState(kv, kv.clone(), zeros(batch, a.heads, 0), mem, mem.clone(),
                *(vec.clone() for _ in range(4)), scalar.clone(), vec.clone(), vec.clone(),
                scalar.clone(), torch.full_like(scalar, -torch.inf)))
        return BlockState(tuple(states), 0, self.capacity, self.block_size)

    def _validate_state(self, state, batch, device):
        if (not isinstance(state.position, int) or state.position < 0 or state.capacity != self.capacity
                or state.block_size != self.block_size or len(state.layers) != len(self.memory)):
            raise ValueError("invalid block state metadata")
        count = min(state.position, self.capacity)
        slots = 3 * (max(0, state.position - self.capacity) // self.block_size)
        for layer, s in zip(self.base.model.layers, state.layers):
            a = layer.self_attn
            vec, scalar = (batch, a.heads, a.head_dim), (batch, a.heads, 1)
            shapes = [(batch, a.kv_heads, count, a.head_dim)] * 2 + [(batch, a.heads, count)]
            shapes += [(batch, a.heads, slots, a.head_dim)] * 2
            shapes += [vec] * 4 + [scalar, vec, vec, scalar, scalar]
            if any(t.shape != shape or t.device != device or t.dtype != torch.float32
                   for t, shape in zip(s.tensors(), shapes)):
                raise ValueError("invalid block state tensor")

    def _layer(self, layer, bank, x, saved, position, read_memory, read_null,
               zero_memory_values, role_mask):
        a = layer.self_attn
        batch, length, _ = x.shape
        z = layer.input_layernorm(x)
        raw_q, new_k, new_v = [p(z).view(batch, length, -1, a.head_dim).transpose(1, 2)
                              for p in (a.q_proj, a.k_proj, a.v_proj)]
        old = saved.key.shape[2]
        keys, values = torch.cat((saved.key, new_k), 2), torch.cat((saved.value, new_v), 2)
        q = _rotate(raw_q, torch.arange(position, position + length, device=x.device), a.rope_theta)
        k = _rotate(keys, torch.arange(position - old, position + length, device=x.device), a.rope_theta)
        groups = a.heads // a.kv_heads
        importance = saved.importance
        outputs, stats = [], []
        for t in range(length):
            right = old + t + 1
            left = max(0, right - self.capacity)
            if right > self.capacity:
                # One lifetime score leaves with its physical token, before the current read.
                evicted = position + t - self.capacity
                saved = bank.accumulate(saved, keys[:, :, left - 1].repeat_interleave(groups, 1),
                    values[:, :, left - 1].repeat_interleave(groups, 1), importance[:, :, 0],
                    evicted % self.block_size + 1, self.block_size)
                importance = importance[:, :, 1:]
            pk = k[:, :, left:right].repeat_interleave(groups, 1)
            pv = values[:, :, left:right].repeat_interleave(groups, 1)
            physical = (q[:, :, t, None] * pk).sum(-1) * a.head_dim ** -.5
            null = (q[:, :, t] * bank.null_weight).sum(-1, keepdim=True) + bank.null_bias[None, :, None]
            if not read_null:
                null = torch.full_like(null, -torch.inf)
            mk, mv = saved.memory_key, saved.memory_value
            memory = (raw_q[:, :, t, None] * mk).sum(-1) * a.head_dim ** -.5
            memory = memory + bank.role_bias.repeat(1, mk.shape[2] // 3)[None]
            if not read_memory:
                memory = torch.full_like(memory, -torch.inf)
            elif role_mask is not None:
                mask = torch.tensor(role_mask, device=x.device, dtype=torch.bool).repeat(mk.shape[2] // 3)
                memory = memory.masked_fill(~mask, -torch.inf)
            probability = torch.cat((physical, null, memory), -1).softmax(-1)
            count = right - left
            received = probability[:, :, :count]
            importance = torch.cat((importance, torch.zeros_like(received[:, :, :1])), -1) + received
            out = (received[..., None] * pv).sum(2)
            if not zero_memory_values:
                out = out + (probability[:, :, count + 1:, None] * mv).sum(2)
            outputs.append(out)
            stats.append(torch.cat((probability[:, :, count:count + 1],
                probability[:, :, count + 1:].reshape(batch, a.heads, -1, 3).sum(2)), -1))
        attended = torch.stack(outputs, 2).transpose(1, 2).reshape(batch, length, -1)
        x = x + a.o_proj(attended)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        result = BlockLayerState(keys[:, :, -self.capacity:].clone(), values[:, :, -self.capacity:].clone(),
                                importance.clone(), *saved.tensors()[3:])
        return (x, *result.tensors(), torch.stack(stats, 2))

    def forward(self, tokens, state=None, *, read_memory=True, read_null=True,
                zero_memory_values=False, role_mask=None, diagnostics=False):
        if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype != torch.long:
            raise ValueError("requires nonempty unpadded int64 (B,T)")
        if role_mask is not None and (len(role_mask) != 3 or any(type(v) is not bool for v in role_mask)):
            raise ValueError("role_mask must contain three booleans")
        weight = self.base.model.embed_tokens.weight
        if weight.dtype != torch.float32 or tokens.device != weight.device:
            raise ValueError("requires FP32 and matching devices")
        if any(p.dtype != torch.float32 for p in self.memory.parameters()):
            raise ValueError("requires FP32 memory")
        state = self.initial_state(tokens.shape[0]) if state is None else state
        self._validate_state(state, tokens.shape[0], tokens.device)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = self.base.model.embed_tokens(tokens)
            layers, masses = [], []
            for layer, bank, saved in zip(self.base.model.layers, self.memory, state.layers):
                def run(hidden, *tensors, layer=layer, bank=bank):
                    return self._layer(layer, bank, hidden, BlockLayerState(*tensors), state.position,
                        read_memory, read_null, zero_memory_values, role_mask)
                if self.checkpointing and self.training and torch.is_grad_enabled():
                    if not x.requires_grad:
                        x = x.requires_grad_(True)
                    result = checkpoint(run, x, *saved.tensors(), use_reentrant=True)
                else:
                    result = run(x, *saved.tensors())
                x = result[0]
                layers.append(BlockLayerState(*result[1:-1]))
                if diagnostics:
                    masses.append(result[-1].detach())
            return BlockOutput(self.base.model.norm(x), BlockState(tuple(layers),
                state.position + tokens.shape[1], self.capacity, self.block_size),
                torch.stack(masses) if diagnostics else None)

    def step(self, tokens, state=None, **kwargs):
        if tokens.ndim != 1:
            raise ValueError("step requires (B,)")
        output = self(tokens[:, None], state, **kwargs)
        return self.base.lm_head(output.hidden[:, 0]), output.state, output.diagnostics
