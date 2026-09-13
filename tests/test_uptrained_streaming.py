import pytest
import torch
from virtualsink.uptraining.native import NativeLM,set_window
from virtualsink.uptraining.qwen import disable_sinks
from virtualsink.uptraining.streaming_fifo import StreamingDecoder


@pytest.mark.parametrize('kind',['llama','qwen2'])
@pytest.mark.parametrize('window',[None,1,4])
@pytest.mark.parametrize('disabled',[False,True])
def test_stream_parity(kind,window,disabled):
    torch.manual_seed(7)
    m=NativeLM(dict(model_type=kind,hidden_size=24,intermediate_size=48,num_hidden_layers=2,
      num_attention_heads=3,num_key_value_heads=1,vocab_size=31,hidden_act='silu',
      rms_norm_eps=1e-5,rope_theta=10000.,tie_word_embeddings=True)).eval()
    disable_sinks(m,disabled);set_window(m,window)
    tokens=torch.arange(20)
    with torch.no_grad(): ref=m.lm_head(m.model(tokens[None]).last_hidden_state)[0]
    d=StreamingDecoder(m,window)
    for i,t in enumerate(tokens):
        actual,*_=d.step(t)
        torch.testing.assert_close(actual,ref[i],atol=2e-6,rtol=1e-5)
        assert all(k.shape[2]==min(i+1,window or i+1) for k,v in d.cache)
    d=StreamingDecoder(m,4,2)
    for t in tokens[:2]:d.step(t)
    prefix=[(k.clone(),v.clone()) for k,v in d.cache]
    for t in tokens[2:]:d.step(t)
    for (k,v),(pk,pv) in zip(d.cache,prefix):
        assert k.shape[2]==4
        torch.testing.assert_close(k[:,:,:2],pk)
        torch.testing.assert_close(v[:,:,:2],pv)
