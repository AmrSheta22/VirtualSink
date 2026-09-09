import pytest
import torch

from virtualsink.uptraining.qwen import sink_attention
from virtualsink.uptraining.native import NativeLM, set_window
from virtualsink.uptraining.evaluate_sinks import paired_summary


@pytest.mark.parametrize("window", [1, 4, 11, 20])
def test_swa_dense_reference_and_gradients(window):
    torch.manual_seed(19)
    q,k,v = [torch.randn(1,2,11,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    w=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    b=torch.randn(2,dtype=torch.float64,requires_grad=True)
    out,gate=sink_attention(q,k,v,w,b,3,True,window)
    pos=torch.arange(11)
    allowed=(pos[None,:]<=pos[:,None]) & (pos[None,:]>pos[:,None]-window)
    scores=(q@k.transpose(-1,-2)/2).masked_fill(~allowed,-torch.inf)
    sink=(q*w[None,:,None]).sum(-1)+b[None,:,None]
    weights=torch.cat((scores,sink[...,None]),-1).softmax(-1)
    ref=weights[...,:-1]@v
    torch.testing.assert_close(out,ref)
    torch.testing.assert_close(gate,weights[...,:-1].sum(-1))
    args=(q,k,v,w,b)
    actual=torch.autograd.grad(out.square().sum()+gate.sum(),args,retain_graph=True)
    expected=torch.autograd.grad(ref.square().sum()+weights[...,:-1].sum(),args)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e)


def test_swa_excludes_old_keys_and_future_keys():
    q,k,v=[torch.randn(1,2,10,4,requires_grad=True) for _ in range(3)]
    out,_=sink_attention(q,k,v,torch.zeros(2,4),torch.zeros(2),3,True,4)
    out[:,:,7].sum().backward()
    for grad in (k.grad,v.grad):
        assert grad[:,:,:4].count_nonzero()==0
        assert grad[:,:,8:].count_nonzero()==0
        assert grad[:,:,4:8].abs().sum()>0


def test_window_setter_and_paired_sign():
    config=dict(model_type="qwen2",vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,hidden_act="silu",
        rope_theta=10000.,rms_norm_eps=1e-6,tie_word_embeddings=True)
    model=NativeLM(config)
    set_window(model,4)
    assert all(layer.self_attn.window_size==4 for layer in model.model.layers)
    with pytest.raises(ValueError):
        set_window(model,-1)
    result=paired_summary([2.,3.,4.],[3.,4.,5.])
    assert result["disabled_minus_enabled_ce"]==1
    assert result["block_bootstrap_95pct_interval"]==[1.,1.]
