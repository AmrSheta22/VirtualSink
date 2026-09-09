"""No-download native decoder equivalence and runtime estimation checks."""

import pytest
import torch

from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.qwen import disable_sinks, parameter_groups
from virtualsink.uptraining.train import language_loss, estimate_remaining


@pytest.mark.parametrize("architecture", ["qwen2", "llama"])
def test_native_logits_and_gradients_match_transformers(architecture):
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
    config_class, model_class = ((Qwen2Config, Qwen2ForCausalLM) if architecture == "qwen2"
                                else (LlamaConfig, LlamaForCausalLM))
    config = config_class(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0, use_cache=False,
        use_sliding_window=False, sliding_window=None, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    reference = model_class(config).eval()
    native = NativeLM(config.to_dict(), chunk_size=3)
    result = native.load_state_dict(reference.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.endswith(("sink_w", "sink_b")) for k in result.missing_keys)
    disable_sinks(native)
    native.gradient_checkpointing_enable()
    native.train()
    tokens = torch.randint(0, 31, (2, 7))
    actual, _ = language_loss(native, tokens, 3)
    expected = reference(tokens, labels=tokens).loss
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.backward()
    expected.backward()
    for name, param in reference.named_parameters():
        torch.testing.assert_close(dict(native.named_parameters())[name].grad, param.grad, atol=1e-6, rtol=1e-4)


def test_timing_includes_validation_and_saving():
    result = estimate_remaining([10, 20, 30], 100, 25, 10, [5], [2])
    assert result["mean_step_seconds"] == 20
    assert result["remaining_seconds"] > 2000
    lo, hi = result["remaining_observed_range_seconds"]
    assert lo < result["remaining_seconds"] < hi


def test_native_sink_gradients_without_transformers():
    config = dict(model_type="qwen2", vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        hidden_act="silu", rope_theta=10000., rms_norm_eps=1e-6, tie_word_embeddings=True)
    model = NativeLM(config, chunk_size=3)
    parameter_groups(model, "attention")
    model.gradient_checkpointing_enable()
    tokens = torch.randint(0, 31, (2, 7))
    loss, gates = language_loss(model, tokens, 3)
    from virtualsink.uptraining.qwen import auxiliary_loss
    (loss + .01 * auxiliary_loss(gates)[0]).backward()
    assert not model.model.embed_tokens.weight.requires_grad
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert model.model.layers[0].self_attn.sink_w.grad.abs().sum() > 0
