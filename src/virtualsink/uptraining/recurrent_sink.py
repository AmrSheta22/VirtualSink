"""Functional FP32 reference for eviction-written LSTM virtual memories.

Wraps NativeLM without changing its attention modules or existing experiment
backend. State belongs to the caller; forward/checkpoint recomputation never
mutates it. A batch has aligned positions and document boundaries. Reset by
passing state=None. Only FIFO physical caches and absolute RoPE are supported.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .native import NativeLM


@dataclass(frozen=True)
class LayerMemoryState:
    key: torch.Tensor  # unrotated, (B, KVH, C, D)
    value: torch.Tensor
    cell: torch.Tensor  # (B, H, 3, D), FP32
    exposed: torch.Tensor

    def detach(self):
        return LayerMemoryState(*(x.detach() for x in
                                  (self.key, self.value, self.cell, self.exposed)))


@dataclass(frozen=True)
class MemoryState:
    layers: Tuple[LayerMemoryState, ...]
    position: int
    capacity: int

    def detach(self):
        """Cut both physical-cache and recurrent-state graphs at a TBPTT boundary."""
        return MemoryState(tuple(layer.detach() for layer in self.layers),
                           self.position, self.capacity)


@dataclass(frozen=True)
class MemoryOutput:
    hidden: torch.Tensor
    state: MemoryState
    diagnostics: Optional[dict] = None


class MemoryBank(nn.Module):
    """Three independent cells and one null sink per query head.

    Transformation matrices are shared within a layer; gate biases and states
    are per head/slot. This avoids adding hundreds of millions of dense gate
    parameters to Qwen1.5B. The cell receives an evicted raw K/V pair,
    repeated across the query heads belonging to that KV head. Retention is
    measured in eviction writes, not calls, blocks, or optimizer iterations.
    """

    def __init__(self, heads, width, timescales=(512., 4096., 16384.)):
        super().__init__()
        if len(timescales) != 3 or any(not math.isfinite(t) or t <= 1 for t in timescales):
            raise ValueError("require three finite retention timescales greater than one")
        self.heads, self.width = heads, width
        self.write_weight = nn.Parameter(torch.empty(2 * width, 4 * width))
        self.recurrent_weight = nn.Parameter(torch.empty(width, 4 * width))
        self.gate_bias = nn.Parameter(torch.zeros(heads, 3, 4 * width))
        self.read_key = nn.Parameter(torch.empty(width, width))
        self.read_value = nn.Parameter(torch.empty(width, width))
        self.null_weight = nn.Parameter(torch.zeros(heads, width))
        # Conservative provisional biases. Calibrate on train-only tokens before training.
        self.null_bias = nn.Parameter(torch.full((heads,), -20.))
        self.memory_bias = nn.Parameter(torch.full((heads, 3), -20.))
        self.register_buffer("timescales", torch.tensor(timescales, dtype=torch.float32))
        nn.init.normal_(self.write_weight, std=(2 * width) ** -.5)
        nn.init.normal_(self.recurrent_weight, std=width ** -.5)
        nn.init.normal_(self.read_key, std=width ** -.5)
        nn.init.normal_(self.read_value, std=width ** -.5)
        with torch.no_grad():
            retention = torch.exp(-1 / self.timescales)
            self.gate_bias[:, :, :width].copy_(torch.logit(retention)[None, :, None])
            self.gate_bias[:, :, width:2 * width].fill_(-4.)

    def write(self, key, value, cell, exposed, forget_override=None):
        u = torch.cat((key.float(), value.float()), -1)
        u = torch.nn.functional.layer_norm(u, (2 * self.width,))
        gates = ((u @ self.write_weight).unsqueeze(2)
                 + exposed @ self.recurrent_weight
                 + self.gate_bias)
        forget, incoming, output, candidate = gates.chunk(4, -1)
        forget = forget.sigmoid()
        if forget_override is not None:
            forget = torch.full_like(forget, forget_override)
        new_cell = forget * cell + incoming.sigmoid() * candidate.tanh()
        return new_cell, output.sigmoid() * new_cell.tanh()

    def read(self, raw_query, rotated_query, exposed):
        keys = exposed @ self.read_key
        values = exposed @ self.read_value
        memory = torch.einsum("bhd,bhsd->bhs", raw_query.float(), keys) * self.width ** -.5
        memory = memory + self.memory_bias
        # Preserve the original null sink's rotated-query affine convention.
        null = torch.einsum("bhd,hd->bh", rotated_query.float(), self.null_weight) + self.null_bias
        return null, memory, values


def _rotate(tensor, positions, theta):
    width = tensor.shape[-1]
    inv = theta ** (-torch.arange(0, width, 2, device=tensor.device).float() / width)
    phase = positions.float()[:, None] * inv[None, :]
    phase = torch.cat((phase, phase), -1)[None, None]
    left, right = tensor.chunk(2, -1)
    return tensor * phase.cos() + torch.cat((-right, left), -1) * phase.sin()


class RecurrentSinkLM(nn.Module):
    """Layerwise causal scan with explicit bounded KV and long-term memory.

    The pretrained model must be FP32. This deliberately unfused reference
    does not support padding, per-row resets, retained prefixes, or rebasing.
    Calling with a new state resets all batch streams together. For differently
    sized documents use separate calls/states, as the pilot trainer does.
    """

    def __init__(self, base, capacity=512, timescales=(512., 4096., 16384.)):
        super().__init__()
        if not isinstance(base, NativeLM):
            raise TypeError("recurrent reference requires NativeLM")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if any(layer.self_attn.head_dim % 2 for layer in base.model.layers):
            raise ValueError("RoPE requires even head dimensions")
        self.base, self.capacity = base, capacity
        self.memory = nn.ModuleList([MemoryBank(layer.self_attn.heads,
                                                layer.self_attn.head_dim, timescales)
                                     for layer in base.model.layers])
        self.memory.to(base.model.embed_tokens.weight.device)
        self.checkpointing = False

    def get_extra_state(self):
        return {"version": 1, "capacity": self.capacity, "parameterization": "layer_shared",
                "timescales": self.memory[0].timescales.tolist()}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("recurrent architecture/checkpoint mismatch")

    def parameter_groups(self, mode="memory", base_lr=2e-5, memory_lr=1e-4):
        """Explicitly own all new parameters; old static sinks are unused/frozen."""
        if mode not in ("memory", "attention", "full"):
            raise ValueError("mode must be memory, attention or full")
        if any(not math.isfinite(lr) or lr <= 0 for lr in (base_lr, memory_lr)):
            raise ValueError("learning rates must be finite and positive")
        base_parameters = []
        for name, parameter in self.base.named_parameters():
            unused = name.endswith((".sink_w", ".sink_b", ".sink_v"))
            projection = any(f".self_attn.{p}." in name
                             for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
            enabled = not unused and (mode == "full" or (mode == "attention" and projection))
            parameter.requires_grad_(enabled)
            if enabled:
                base_parameters.append(parameter)
        for parameter in self.memory.parameters():
            parameter.requires_grad_(True)
        groups = [{"params": list(self.memory.parameters()), "lr": memory_lr, "weight_decay": 0.}]
        if base_parameters:
            groups.insert(0, {"params": base_parameters, "lr": base_lr, "weight_decay": .01})
        return groups

    def initial_state(self, batch):
        device = self.base.model.embed_tokens.weight.device
        layers = []
        for layer in self.base.model.layers:
            a = layer.self_attn
            kv = torch.zeros(batch, a.kv_heads, 0, a.head_dim, device=device)
            cell = torch.zeros(batch, a.heads, 3, a.head_dim, device=device)
            layers.append(LayerMemoryState(kv, kv.clone(), cell, cell.clone()))
        return MemoryState(tuple(layers), 0, self.capacity)

    def _validate_state(self, state, batch, device):
        if (not isinstance(state.position, int) or state.position < 0
                or state.capacity != self.capacity or len(state.layers) != len(self.memory)):
            raise ValueError("invalid state position, capacity or layer count")
        count = min(state.position, self.capacity)
        for layer, saved in zip(self.base.model.layers, state.layers):
            a = layer.self_attn
            shapes = [(batch, a.kv_heads, count, a.head_dim)] * 2 + [(batch, a.heads, 3, a.head_dim)] * 2
            for tensor, shape in zip((saved.key, saved.value, saved.cell, saved.exposed), shapes):
                if tensor.shape != shape or tensor.device != device or tensor.dtype != torch.float32:
                    raise ValueError("state shape/device/dtype mismatch")

    def _layer(self, layer, bank, x, key, value, cell, exposed, position,
               read_memory, write_memory, read_null, forget_override, zero_memory_values,
               diagnostics, bypass_virtual, memory_teacher):
        a = layer.self_attn
        batch, length, _ = x.shape
        z = layer.input_layernorm(x)
        raw_q, new_k, new_v = [p(z).view(batch, length, -1, a.head_dim).transpose(1, 2)
                              for p in (a.q_proj, a.k_proj, a.v_proj)]
        old_count = key.shape[2]
        keys, values = torch.cat((key, new_k), 2), torch.cat((value, new_v), 2)
        rotated_keys = _rotate(keys, torch.arange(position - old_count, position + length,
                                                   device=x.device), a.rope_theta)
        queries = _rotate(raw_q, torch.arange(position, position + length, device=x.device), a.rope_theta)
        groups = a.heads // a.kv_heads
        outputs = []
        stats = [[], [], [], [], []]
        for t in range(length):
            right = old_count + t + 1
            left = max(0, right - self.capacity)
            if left and write_memory:
                # The one token leaving FIFO at this step, before the current read.
                evicted_key = keys[:, :, left - 1].repeat_interleave(groups, 1)
                evicted_value = values[:, :, left - 1].repeat_interleave(groups, 1)
                cell, exposed = bank.write(evicted_key, evicted_value, cell, exposed, forget_override)
            physical_keys = rotated_keys[:, :, left:right].repeat_interleave(groups, 1)
            physical_values = values[:, :, left:right].repeat_interleave(groups, 1)
            q = queries[:, :, t].float()
            physical = torch.einsum("bhd,bhkd->bhk", q, physical_keys.float()) * a.head_dim ** -.5
            null, memory, memory_values = bank.read(raw_q[:, :, t], q, exposed)
            if not read_null:
                null = torch.full_like(null, -torch.inf)
            if not read_memory:
                memory = torch.full_like(memory, -torch.inf)
            virtual = torch.cat((null.unsqueeze(-1), memory), -1)
            attention_virtual = torch.full_like(virtual, -torch.inf) if bypass_virtual else virtual
            probability = torch.cat((physical, attention_virtual), -1).softmax(-1)
            count = right - left
            if memory_teacher and left:
                # A separate causal teacher trajectory through every native
                # residual/MLP block. Never substitute stale ordinary states.
                probability = torch.cat((torch.zeros_like(physical),
                                         torch.zeros_like(null.unsqueeze(-1)),
                                         memory.softmax(-1)), -1)
            out = torch.einsum("bhk,bhkd->bhd", probability[..., :count], physical_values.float())
            if not zero_memory_values:
                out = out + torch.einsum("bhs,bhsd->bhd", probability[..., count + 1:], memory_values)
            outputs.append(out)
            if diagnostics:
                for collection, entry in zip(stats, (physical.logsumexp(-1), virtual.logsumexp(-1),
                                                      probability[..., count], probability[..., count + 1:].sum(-1))):
                    collection.append(entry.detach())
                # Connected log odds permit stable gate supervision even when
                # the ordinary softmax assigns almost no mass to memory.
                stats[4].append(memory.logsumexp(-1) -
                                torch.cat((physical, null.unsqueeze(-1)), -1).logsumexp(-1))
        attended = torch.stack(outputs, 2).transpose(1, 2).reshape(batch, length, -1)
        x = x + a.o_proj(attended)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Clone even a contiguous slice: a detached cache must not retain the
        # storage of the entire processed segment (notably B=KVH=1).
        saved = (keys[:, :, -self.capacity:].clone(memory_format=torch.contiguous_format),
                 values[:, :, -self.capacity:].clone(memory_format=torch.contiguous_format), cell, exposed)
        collected = tuple(torch.stack(entries, -1) for entries in stats) if diagnostics else ()
        return (x, *saved, *collected)

    def forward(self, tokens, state=None, *, read_memory=True, write_memory=True,
                read_null=True, forget_override=None, zero_memory_values=False,
                diagnostics=False, bypass_virtual=False, memory_teacher=False):
        if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be nonempty int64 (B,T), without padding")
        if memory_teacher and (not read_memory or bypass_virtual):
            raise ValueError("memory teacher requires enabled memory reads")
        weight = self.base.model.embed_tokens.weight
        if weight.dtype != torch.float32 or tokens.device != weight.device:
            raise ValueError("reference requires FP32 model and tokens on its device")
        if forget_override is not None and (not math.isfinite(forget_override) or not 0 <= forget_override <= 1):
            raise ValueError("forget override must lie in [0,1]")
        state = self.initial_state(tokens.shape[0]) if state is None else state
        self._validate_state(state, tokens.shape[0], tokens.device)
        if any(p.dtype != torch.float32 for p in self.memory.parameters()):
            raise ValueError("memory parameters must be FP32")
        # FP32 cell/gate arithmetic also under an enclosing autocast context.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = self.base.model.embed_tokens(tokens)
            next_layers, collected = [], []
            for layer, bank, saved in zip(self.base.model.layers, self.memory, state.layers):
                def run(hidden, key, value, cell, exposed, layer=layer, bank=bank):
                    return self._layer(layer, bank, hidden, key, value, cell, exposed, state.position,
                                       read_memory, write_memory, read_null, forget_override,
                                       zero_memory_values, diagnostics, bypass_virtual, memory_teacher)
                args = (x, saved.key, saved.value, saved.cell, saved.exposed)
                if self.checkpointing and self.training and torch.is_grad_enabled():
                    if not x.requires_grad:
                        x = x.requires_grad_(True)
                        args = (x, *args[1:])
                    result = checkpoint(run, *args, use_reentrant=True)
                else:
                    result = run(*args)
                x = result[0]
                next_layers.append(LayerMemoryState(*result[1:5]))
                if diagnostics:
                    collected.append(result[5:])
            info = None
            if diagnostics:
                names = ("physical_log_partition", "virtual_log_partition", "null_mass", "memory_mass", "memory_gate_logit")
                info = {name: torch.stack([entry[i] for entry in collected]) for i, name in enumerate(names)}
            return MemoryOutput(self.base.model.norm(x),
                                MemoryState(tuple(next_layers), state.position + tokens.shape[1], self.capacity), info)

    def step(self, tokens, state=None, **kwargs):
        """Single-token API; differentiable unless the caller uses no_grad."""
        if tokens.ndim != 1:
            raise ValueError("step tokens must have shape (B,)")
        result = self(tokens[:, None], state, **kwargs)
        return self.base.lm_head(result.hidden[:, 0]), result.state, result.diagnostics

    @torch.no_grad()
    def calibrate(self, documents, target_mass=1e-4, quantile=1., passes=3, flags=None):
        """Calibrate common virtual-bias shifts on a small train-only sample.

        Each (B,T) document starts from fresh state and must include eviction.
        First measure prospective scores while physical attention follows the
        unchanged base model. Then refine with enabled virtual reads. The default
        targets the largest observed total virtual mass per layer/head; this is
        a measured initialization heuristic, not a bound on unseen inputs.
        """
        if not 0 < target_mass < .1 or not 0 < quantile <= 1 or passes < 1:
            raise ValueError("invalid calibration target, quantile or passes")
        documents = tuple(documents)
        if not documents or any(d.ndim != 2 or d.shape[1] <= self.capacity for d in documents):
            raise ValueError("calibration requires documents longer than the physical window")
        target_logit = math.log(target_mass / (1 - target_mass))
        flags = {} if flags is None else dict(flags)
        if not flags.get("read_null", True) and not flags.get("read_memory", True):
            raise ValueError("cannot calibrate disabled virtual attention")
        training = self.training
        self.eval()
        try:
            for iteration in range(passes):
                deltas = []
                for tokens in documents:
                    info = self(tokens, diagnostics=True, bypass_virtual=(iteration == 0), **flags).diagnostics
                    delta = info["virtual_log_partition"] - info["physical_log_partition"]
                    deltas.append(delta.permute(0, 2, 1, 3).flatten(2))
                # Select an observed order statistic. Linear interpolation in
                # logit space would not equal interpolation in probability space.
                reference = torch.quantile(torch.cat(deltas, -1), quantile, dim=-1, interpolation="nearest")
                for bank, correction in zip(self.memory, target_logit - reference):
                    bank.null_bias.add_(correction)
                    bank.memory_bias.add_(correction[:, None])
            samples = []
            for tokens in documents:
                info = self(tokens, diagnostics=True, **flags).diagnostics
                samples.append((info["null_mass"] + info["memory_mass"]).permute(0, 2, 1, 3).flatten(2))
            mass = torch.cat(samples, -1)
            return {"target_mass": target_mass, "quantile": quantile,
                    "mean_mass": mass.mean(-1).cpu().tolist(),
                    "quantile_mass": torch.quantile(mass, quantile, dim=-1, interpolation="nearest").cpu().tolist(),
                    "max_mass": mass.max(-1).values.cpu().tolist()}
        finally:
            self.train(training)
