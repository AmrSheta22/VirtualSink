import math
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtualsink.uptraining.prefix_kv import attend,rotate


class Rotary(torch.nn.Module):
    def forward(self,tensor,seq_len):
        phase=torch.arange(seq_len,device=tensor.device).float()[:,None]*torch.tensor([1.,.01],device=tensor.device)[None]
        phase=torch.cat((phase,phase),-1)[None,None]
        return phase.cos(),phase.sin()


def attention(capacity=8):
    torch.manual_seed(17)
    return SimpleNamespace(rotary_emb=Rotary(),rotary_ndims=4,norm_factor=1/math.sqrt(8),
        virtual_keys=torch.nn.Parameter(torch.randn(2,4,8)),
        virtual_values=torch.nn.Parameter(torch.randn(2,4,8)),virtual_capacity=capacity)


def test_parallel_eviction_and_positions_match_explicit_rolling_joint_attention():
    a=attention();q=torch.randn(1,2,19,8);k=torch.randn_like(q);v=torch.randn_like(q);positions=torch.arange(19)
    full,_=attend(a,q,k,v,positions,positions)
    for i in range(19):
        start=max(0,i-4);length=i-start+1
        local=torch.arange(length);qr=rotate(a,q[:,:,i:i+1],local[-1:]+4)
        keys=rotate(a,torch.cat((a.virtual_keys[None],k[:,:,start:i+1]),2),torch.arange(length+4))
        values=torch.cat((a.virtual_values[None],v[:,:,start:i+1]),2)
        expected=((qr@keys.transpose(-1,-2))*a.norm_factor).softmax(-1)@values
        torch.testing.assert_close(full[:,:,i:i+1],expected,atol=1e-6,rtol=1e-5)


def test_key_and_nonzero_value_gradients_reach_every_slot():
    a=attention();q=torch.randn(1,2,13,8);k=torch.randn_like(q);v=torch.randn_like(q);positions=torch.arange(13)
    out,_=attend(a,q,k,v,positions,positions);out.square().sum().backward()
    for p in [a.virtual_keys,a.virtual_values]:
        assert torch.isfinite(p.grad).all()
        assert (p.grad.norm(dim=-1)>0).all()


def test_virtual_value_content_changes_output_and_future_queries_are_masked():
    a=attention();q=torch.randn(1,2,12,8);k=torch.randn_like(q);v=torch.randn_like(q);indices=torch.arange(12)
    out,w=attend(a,q,k,v,indices,indices)
    assert (w[:,:,0,5:]==0).all()  #4memory slots + first physical key only
    assert (w[:,:,-1,4:11]==0).all()  #real keys0..6 were evicted
    with torch.no_grad():a.virtual_values.zero_()
    zero,_=attend(a,q,k,v,indices,indices)
    assert not torch.allclose(out,zero)
