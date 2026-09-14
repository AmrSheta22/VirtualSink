import pytest
import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.qwen import disable_sinks
from virtualsink.uptraining.streaming_fifo import StreamingDecoder

@pytest.mark.parametrize('prefix',[0,2])
def test_rebase(prefix):
    torch.manual_seed(7)
    m=NativeLM(dict(model_type='llama',hidden_size=24,intermediate_size=48,num_hidden_layers=2,
      num_attention_heads=3,num_key_value_heads=1,vocab_size=31,hidden_act='silu',
      rms_norm_eps=1e-5,rope_theta=10000.,tie_word_embeddings=True)).eval()
    disable_sinks(m)
    absolute=StreamingDecoder(m,4,prefix)
    rebased=StreamingDecoder(m,4,prefix,rebase=True)
    for i in range(20):
        a,*_=absolute.step(torch.tensor(i)); b,*_=rebased.step(torch.tensor(i))
        if i<4 or prefix==0: torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5)
        assert all(k.shape[2]<=4 for k,v in rebased.cache)
        if i==1: stored=[k.clone() for k,v in rebased.cache]
    if prefix:
        for (k,v),initial in zip(rebased.cache,stored):torch.testing.assert_close(k[:,:,:2],initial)
