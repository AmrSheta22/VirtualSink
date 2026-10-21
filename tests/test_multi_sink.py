import pytest
import torch
from virtualsink.uptraining.qwen import sink_attention, auxiliary_loss, parameter_groups
from virtualsink.uptraining.native import NativeLM, configure_sinks, virtual_values, set_window
from virtualsink.uptraining.streaming_fifo import StreamingDecoder
from virtualsink.uptraining.sink_validation import check_value_gradients


@pytest.mark.parametrize('recompute', [False, True, 'reentrant'])
def test_four_sink_dense_output_and_gradient(recompute):
    torch.manual_seed(81)
    q,k,v = [torch.randn(1,2,9,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    w = torch.randn(2,4,4,dtype=torch.float64,requires_grad=True)
    b = torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    sv = torch.randn(2,4,4,dtype=torch.float64,requires_grad=True)
    out,gate = sink_attention(q,k,v,w,b,3,recompute,4,sv)
    pos = torch.arange(9)
    mask = (pos[None,:] <= pos[:,None]) & (pos[None,:] > pos[:,None]-4)
    physical = (q@k.transpose(-1,-2)/2).masked_fill(~mask,-torch.inf)
    scores = q@w.transpose(-1,-2)+b[None,:,None,:]
    prob = torch.cat((physical,scores),-1).softmax(-1)
    ref = prob[...,:9]@v+prob[...,9:]@sv
    torch.testing.assert_close(out,ref)
    torch.testing.assert_close(gate,prob[...,:9].sum(-1))
    params = (q,k,v,w,b,sv)
    (out.square().sum()+gate.sum()).backward(retain_graph=True)
    actual = [p.grad.clone() for p in params]
    expected = torch.autograd.grad(ref.square().sum()+prob[...,:9].sum(),params)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e)


def test_four_sink_initialization_optimizer_and_eviction_parity():
    torch.manual_seed(9)
    config = dict(model_type='qwen2',vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,hidden_act='silu',
        rope_theta=10000.,rms_norm_eps=1e-6,tie_word_embeddings=True)
    model = NativeLM(config,3)
    configure_sinks(model,4,1e-9)
    groups = parameter_groups(model,base_lr=1e-4,sink_lr=5e-4)
    for layer in model.model.layers:
        a = layer.self_attn
        assert virtual_values(a)[:,0].count_nonzero() == 0
        rms = a.v_proj.weight.square().mean().sqrt()
        torch.testing.assert_close(a.sink_v.square().mean(-1).sqrt(),
            (rms*torch.tensor([1/3,2/3,1.])*1e-9).expand(a.heads,-1),atol=1e-17,rtol=1e-5)
        assert any(p is a.sink_v for p in groups[1]['params'])
    set_window(model,4)
    tokens = torch.randint(0,31,(1,17))
    model.train(); model.gradient_checkpointing_enable()
    result = model.model(tokens)
    aux,_ = auxiliary_loss(result.attentions)
    aux.backward()
    assert all(l.self_attn.sink_b.grad.abs().sum()>0 for l in model.model.layers)
    # Make values appreciable so parity exercises their contribution, not rounding-to-zero.
    with torch.no_grad():
        for l in model.model.layers:
            l.self_attn.sink_v.fill_(.1)
    model.eval()
    with torch.no_grad():
        expected = model.lm_head(model.model(tokens).last_hidden_state)[0]
        decoder = StreamingDecoder(model,4)
        for i,t in enumerate(tokens[0]):
            actual,*_ = decoder.step(t)
            torch.testing.assert_close(actual,expected[i],atol=1e-5,rtol=1e-5)
        assert all(k.shape[2]==4 for k,v in decoder.cache)


def test_saturated_head_zero_language_gradient_is_not_disconnection():
    from types import SimpleNamespace
    a = SimpleNamespace(heads=2,head_dim=4,
        sink_w=torch.nn.Parameter(torch.zeros(2,4,4)),
        sink_b=torch.nn.Parameter(torch.full((2,4),-3.5)),
        sink_v=torch.nn.Parameter(torch.full((2,3,4),1e-10)))
    q = torch.full((1,2,8,4),100.)
    # Physical logits are 20,000, so all virtual probabilities underflow to zero.
    out,_ = sink_attention(q,q,torch.zeros_like(q),a.sink_w,a.sink_b,8,False,8,virtual_values(a))
    out.sum().backward()
    assert a.sink_v.grad is not None and a.sink_v.grad.count_nonzero()==0
    result = check_value_gradients(a)
    assert len(result['zero_language_gradient_head_slots'])==6
    assert result['controlled_gradient_min']>0
    a.sink_v.grad=None
    with pytest.raises(AssertionError,match='connected'):
        check_value_gradients(a)


def test_missing_streaming_checkpoint_identity_reports_incomplete(tmp_path):
    import json
    from virtualsink.uptraining.summarize_qwen15_ablations import summarize
    with pytest.raises(ValueError,match='identity unavailable'):
        summarize(tmp_path)
    assert json.loads((tmp_path/'INCOMPLETE.json').read_text())['missing']==['checkpoint-identity.json']
