"""Teacher-forced streaming with absolute RoPE positions and bounded per-layer KV.

Evaluation only. Stored KV has at most capacity entries, including the current
token. The virtual sink is a scalar logit, not a retained physical cache slot.
"""
import torch


class StreamingDecoder:
    def __init__(self, model, capacity=None, retain_prefix=0, rebase=False):
        if capacity is not None and (capacity < 1 or not 0 <= retain_prefix < capacity):
            raise ValueError("invalid cache capacity/prefix")
        self.model = model.eval()
        self.capacity, self.retain_prefix = capacity, retain_prefix
        self.cache = [None] * len(model.model.layers)
        self.position = 0
        self.rebase = rebase

    @torch.no_grad()
    def step(self, token):
        x = self.model.model.embed_tokens(token.reshape(1, 1))
        masses, sink_logits, physical_lse = [], [], []
        for index, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            z = layer.input_layernorm(x)
            q, k, v = [p(z).view(1, 1, -1, attn.head_dim).transpose(1, 2)
                       for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
            inv = 1 / attn.rope_theta ** (torch.arange(0, attn.head_dim, 2, device=x.device).float() / attn.head_dim)
            phase = torch.cat((inv, inv)) * self.position
            def rotate(t):
                half = attn.head_dim // 2
                return t * phase.cos() + torch.cat((-t[..., half:], t[..., :half]), -1) * phase.sin()
            if not self.rebase:
                q, k = rotate(q), rotate(k)
            previous = self.cache[index]
            if previous is not None:
                k, v = torch.cat((previous[0], k), 2), torch.cat((previous[1], v), 2)
            if self.capacity is not None and k.shape[2] > self.capacity:
                recent = self.capacity - self.retain_prefix
                k = torch.cat((k[:, :, :self.retain_prefix], k[:, :, -recent:]), 2)
                v = torch.cat((v[:, :, :self.retain_prefix], v[:, :, -recent:]), 2)
            self.cache[index] = k, v
            if self.rebase:
                # Cache unrotated keys; apply RoPE at their current cache indices.
                phase = torch.cat((inv, inv)) * (k.shape[2]-1)
                q = rotate(q)
                phase = torch.arange(k.shape[2],device=x.device)[:,None] * torch.cat((inv,inv))[None,:]
                k = rotate(k)
            groups = attn.heads // attn.kv_heads
            keys, values = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
            scores = (q.float() @ keys.float().transpose(-1, -2)) * attn.head_dim ** -.5
            sink = (q.float() * attn.sink_w[None, :, None]).sum(-1) + attn.sink_b[None, :, None]
            if attn.sink_disabled:
                sink = torch.full_like(sink, -1e6)
            weights = torch.cat((scores, sink[..., None]), -1).softmax(-1)
            out = weights[..., :-1] @ values.float()
            x = x + attn.o_proj(out.transpose(1, 2).reshape(1, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            masses.append(weights[0, :, 0, -1])
            sink_logits.append(sink[0, :, 0])
            physical_lse.append(scores.logsumexp(-1)[0, :, 0])
        self.position += 1
        logits = self.model.lm_head(self.model.model.norm(x))[0, 0]
        return logits, torch.stack(masses), torch.stack(sink_logits), torch.stack(physical_lse)
