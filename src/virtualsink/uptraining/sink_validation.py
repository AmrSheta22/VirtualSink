"""Value-gradient checks that distinguish disconnected parameters from saturated heads."""
import torch
from .native import virtual_values
from .qwen import sink_attention


def check_value_gradients(attn):
    value = attn.sink_v
    if not value.requires_grad or value.grad is None or not torch.isfinite(value.grad).all():
        raise AssertionError('sink values must have connected, finite language-loss gradients')
    real = value.grad.detach().abs().sum(-1)
    # A head can assign zero virtual mass in FP32. Test trainability separately
    # with moderate physical logits and a known nonzero upstream gradient.
    q = value.new_zeros(1,attn.heads,8,attn.head_dim)
    out,_ = sink_attention(q,torch.zeros_like(q),torch.zeros_like(q),attn.sink_w,
                          attn.sink_b,8,False,8,virtual_values(attn))
    probe = torch.autograd.grad(out.sum(),value)[0].abs().sum(-1)
    if not torch.isfinite(probe).all() or not (probe > 0).all():
        raise AssertionError('controlled probe failed to reach every trainable value vector')
    return {'zero_language_gradient_head_slots':(real==0).nonzero().cpu().tolist(),
            'language_gradient_l1_sum':real.sum().item(),
            'controlled_gradient_min':probe.min().item()}
