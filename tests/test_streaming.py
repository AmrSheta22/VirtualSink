import copy
import torch
import pytest
from virtualsink.pretraining.recipe import initialize
from virtualsink.pretraining.streaming import StreamingDecoder
from virtualsink.uptraining.native import set_window
from virtualsink.uptraining.qwen import disable_sinks


def tiny():
    return dict(model_type="llama",hidden_size=24,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,
        max_position_embeddings=32,vocab_size=31,hidden_act="silu",
        rms_norm_eps=1e-5,rope_theta=10000.,tie_word_embeddings=True)


@pytest.mark.parametrize("window",[None,1,4])
@pytest.mark.parametrize("disabled",[False,True])
def test_cached_logits_match_parallel_across_eviction(window,disabled):
    model=initialize(8,3,tiny(),sink_weight_std=.02,sink_bias=4.).eval()
    disable_sinks(model,disabled); set_window(model,window)
    tokens=torch.arange(20)[None]
    with torch.no_grad(): expected=model.lm_head(model.model(tokens).last_hidden_state)[0]
    decoder=StreamingDecoder(model,window)
    for i,token in enumerate(tokens[0]):
        actual,*_=decoder.step(token)
        torch.testing.assert_close(actual,expected[i],atol=1e-6,rtol=1e-5)
        for k,v in decoder.cache:
            assert k.shape[2]==min(i+1,window or i+1)


def test_prefix_retained_with_same_total_budget():
    model=initialize(8,3,tiny()).eval()
    decoder=StreamingDecoder(model,4,2)
    decoder.step(torch.tensor(0)); decoder.step(torch.tensor(1))
    prefix=[(k.clone(),v.clone()) for k,v in decoder.cache]
    for i in range(2,15): decoder.step(torch.tensor(i))
    for (k,v),(pk,pv) in zip(decoder.cache,prefix):
        assert k.shape[2]==4
        torch.testing.assert_close(k[:,:,:2],pk)
        torch.testing.assert_close(v[:,:,:2],pv)


def test_full_attention_nested_checkpoint_gradients():
    from virtualsink.pretraining.train import language_loss
    a=initialize(8,3,tiny())
    b=copy.deepcopy(a); b.gradient_checkpointing_enable()
    tokens=torch.arange(13)[None]
    language_loss(a,tokens,3)[0].backward()
    language_loss(b,tokens,3)[0].backward()
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_diagnostic_outputs(tmp_path):
    from virtualsink.pretraining.diagnose_streaming import evaluate_stream,heatmap,atomic_json,write_report
    model=initialize(8,3,tiny()).eval()
    result=evaluate_stream(model,torch.arange(20),4,0,4)
    assert result["max_cache_entries"]==4
    assert len(result["per_token_ce"])==19
    assert len(result["bins"])==5
    atomic_json(tmp_path/"result-standard--0--full.json",result)
    atomic_json(tmp_path/"result-standard--0--fifo.json",result)
    heatmap(tmp_path/"heads.svg",result["mean_virtual_mass_by_layer_head"],"Test")
    write_report(tmp_path)
    assert (tmp_path/"REPORT.md").exists()
    assert (tmp_path/"loss_by_position.svg").exists()
