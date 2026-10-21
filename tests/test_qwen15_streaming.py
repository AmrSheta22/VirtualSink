import pytest
import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.qwen import disable_sinks
from virtualsink.uptraining.streaming_fifo import StreamingDecoder
from virtualsink.uptraining.rebased_reference import RebasedReference
from virtualsink.uptraining.summarize_qwen15_streaming import token_scores


@pytest.mark.parametrize("prefix", [0, 4])
def test_qwen_rebase_matches_absolute_key_delta_oracle_after_repeated_eviction(prefix):
    torch.manual_seed(31)
    model = NativeLM(dict(model_type="qwen2", hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=12, num_key_value_heads=2,
        vocab_size=41, hidden_act="silu", rms_norm_eps=1e-6, rope_theta=1e6,
        tie_word_embeddings=True)).eval()
    disable_sinks(model)
    actual = StreamingDecoder(model, 8, prefix, rebase=True)
    reference = RebasedReference(model, 8, prefix)
    for position in range(40):
        token = torch.tensor(position)
        logits, *_ = actual.step(token)
        torch.testing.assert_close(logits, reference.step(token), atol=4e-6, rtol=2e-5)
        assert all(key.shape[2] == min(position + 1, 8) for key, value in actual.cache)


def test_statistics_use_exact_eviction_boundary_and_reject_missing_nonfinite_losses():
    losses = [1.] * 512 + [2.] * (8192 - 512)
    assert token_scores({"per_token_ce": losses}) == {"pre_eviction": 1., "post_eviction": 2., "beyond_10w": 2.}
    with pytest.raises(ValueError):
        token_scores({"per_token_ce": losses[:-1]})
    losses[-1] = float("nan")
    with pytest.raises(ValueError):
        token_scores({"per_token_ce": losses})
