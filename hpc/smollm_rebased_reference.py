"""Independent evaluation oracle: store absolute-rotated keys, then rotate by delta.

Only standard attention. Production rebasing instead caches unrotated keys.
Hidden states remain cached; this does not recompute historical token states.
"""
import torch


class RebasedReference:
    def __init__(self, model, capacity=512, prefix=4):
        if not 0 <= prefix < capacity:
            raise ValueError("invalid capacity/prefix")
        if not all(layer.self_attn.sink_disabled for layer in model.model.layers):
            raise ValueError("reference requires standard attention")
        self.model = model.eval()
        self.capacity, self.prefix, self.position = capacity, prefix, 0
        self.cache = [None] * len(model.model.layers)

    @torch.no_grad()
    def step(self, token):
        x = self.model.model.embed_tokens(token.reshape(1, 1))
        for index, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            hidden = layer.input_layernorm(x)
            query, key, value = [projection(hidden).reshape(1, 1, -1, attention.head_dim).transpose(1, 2)
                                 for projection in [attention.q_proj, attention.k_proj, attention.v_proj]]
            inverse = attention.rope_theta ** (-torch.arange(0, attention.head_dim, 2, device=x.device).float() / attention.head_dim)
            def rotate(tensor, positions):
                angle = positions.float().reshape(-1, 1) * inverse.reshape(1, -1)
                angle = torch.cat([angle, angle], -1).reshape(1, 1, -1, attention.head_dim)
                left, right = tensor.chunk(2, -1)
                return tensor * angle.cos() + torch.cat([-right, left], -1) * angle.sin()
            positions = torch.tensor([self.position], device=x.device)
            key = rotate(key, positions)
            if self.cache[index] is not None:
                prior_key, prior_value, prior_positions = self.cache[index]
                key = torch.cat([prior_key, key], 2)
                value = torch.cat([prior_value, value], 2)
                positions = torch.cat([prior_positions, positions])
            if len(positions) > self.capacity:
                selection = torch.cat([torch.arange(self.prefix, device=x.device),
                    torch.arange(len(positions) - (self.capacity - self.prefix), len(positions), device=x.device)])
                key, value = key.index_select(2, selection), value.index_select(2, selection)
                positions = positions.index_select(0, selection)
            self.cache[index] = key, value, positions
            query = rotate(query, torch.tensor([len(positions) - 1], device=x.device))
            relative_key = rotate(key, torch.arange(len(positions), device=x.device) - positions)
            groups = attention.heads // attention.kv_heads
            repeated_key = relative_key.repeat_interleave(groups, 1)
            repeated_value = value.repeat_interleave(groups, 1)
            logits = query @ repeated_key.transpose(-1, -2) / attention.head_dim ** .5
            output = logits.softmax(-1) @ repeated_value
            x = x + attention.o_proj(output.transpose(1, 2).reshape(1, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.position += 1
        return self.model.lm_head(self.model.model.norm(x))[0, 0]
