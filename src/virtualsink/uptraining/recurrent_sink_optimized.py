"""Batched FP32 implementation of the reference recurrence.

The LSTM scan remains sequential. Attention uses a masked segment matrix,
so its temporary score storage is O(T*(T+C)), unlike the reference loop.
"""
import torch
import torch.nn.functional as F

from .recurrent_sink import RecurrentSinkLM, _rotate


class OptimizedRecurrentSinkLM(RecurrentSinkLM):
    def _layer(self, layer, bank, x, key, value, cell, exposed, position,
               read_memory, write_memory, read_null, forget_override,
               zero_memory_values, diagnostics, bypass_virtual, memory_teacher):
        a = layer.self_attn
        batch, length, _ = x.shape
        z = layer.input_layernorm(x)
        raw_q, new_k, new_v = [p(z).view(batch, length, -1, a.head_dim).transpose(1, 2)
                              for p in (a.q_proj, a.k_proj, a.v_proj)]
        old = key.shape[2]
        keys, values = torch.cat((key, new_k), 2), torch.cat((value, new_v), 2)
        groups = a.heads // a.kv_heads
        q = _rotate(raw_q, torch.arange(position, position + length, device=x.device), a.rope_theta)
        k = _rotate(keys, torch.arange(position-old, position+length, device=x.device), a.rope_theta)
        k = k.repeat_interleave(groups, 1)
        v = values.repeat_interleave(groups, 1)
        # Only the exposed-state projection depends on the recurrent scan.
        writes = max(0, old + length - self.capacity)
        projected = None
        if write_memory and writes:
            u = torch.cat((keys[:, :, :writes], values[:, :, :writes]), -1)
            projected = (F.layer_norm(u, (2*a.head_dim,)) @ bank.write_weight).repeat_interleave(groups, 1)
        history = []
        for t in range(length):
            left = max(0, old+t+1-self.capacity)
            if write_memory and left:
                gates = projected[:, :, left-1].unsqueeze(2) + exposed @ bank.recurrent_weight + bank.gate_bias
                forget, incoming, output, candidate = gates.chunk(4, -1)
                forget = forget.sigmoid()
                if forget_override is not None:
                    forget = torch.full_like(forget, forget_override)
                cell = forget*cell + incoming.sigmoid()*candidate.tanh()
                exposed = output.sigmoid()*cell.tanh()
            history.append(exposed)
        states = torch.stack(history, 2)
        memory_keys, memory_values = states @ bank.read_key, states @ bank.read_value
        memory = (raw_q.unsqueeze(-2)*memory_keys).sum(-1)*a.head_dim**-.5 + bank.memory_bias[None, :, None]
        null = (q*bank.null_weight[None, :, None]).sum(-1) + bank.null_bias[None, :, None]
        physical = (q @ k.transpose(-1, -2))*a.head_dim**-.5
        columns = torch.arange(old+length, device=x.device)[None, :]
        right = old+torch.arange(length, device=x.device)[:, None]+1
        valid = (columns < right) & (columns >= (right-self.capacity).clamp_min(0))
        physical = physical.masked_fill(~valid[None, None], -torch.inf)
        if not read_null:
            null = torch.full_like(null, -torch.inf)
        if not read_memory:
            memory = torch.full_like(memory, -torch.inf)
        virtual = torch.cat((null.unsqueeze(-1), memory), -1)
        av = torch.full_like(virtual, -torch.inf) if bypass_virtual else virtual
        probability = torch.cat((physical, av), -1).softmax(-1)
        count = old+length
        if memory_teacher:
            teacher = torch.cat((torch.zeros_like(physical), torch.zeros_like(null.unsqueeze(-1)), memory.softmax(-1)), -1)
            probability = torch.where((right[:, 0] > self.capacity)[None, None, :, None], teacher, probability)
        attended = probability[..., :count] @ v
        if not zero_memory_values:
            attended = attended + (probability[..., count+1:, None]*memory_values).sum(-2)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        x = x + a.o_proj(attended)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        saved = (keys[:, :, -self.capacity:].clone(memory_format=torch.contiguous_format),
                 values[:, :, -self.capacity:].clone(memory_format=torch.contiguous_format), cell, exposed)
        stats = ()
        if diagnostics:
            stats = (physical.logsumexp(-1).detach(), virtual.logsumexp(-1).detach(),
                     probability[..., count].detach(), probability[..., count+1:].sum(-1).detach(),
                     memory.logsumexp(-1)-torch.cat((physical, null.unsqueeze(-1)), -1).logsumexp(-1))
        return (x, *saved, *stats)
