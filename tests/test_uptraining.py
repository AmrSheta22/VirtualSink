"""CPU numerical checks; no pretrained downloads or optimization runs."""

import copy

import numpy as np
import pytest
import torch

from virtualsink.uptraining.qwen import sink_attention, auxiliary_loss
from virtualsink.uptraining.data import TokenBlocks


def test_chunked_attention_matches_dense_forward_backward():
    torch.manual_seed(7)
    tensors = [torch.randn(2, 4, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    w = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    q, k, v = tensors
    actual, gates = sink_attention(q, k, v, w, b, 3)
    scores = (q @ k.transpose(-1, -2) / 2).masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
    sink = (q * w[None, :, None]).sum(-1) + b[None, :, None]
    weights = torch.cat((scores, sink[..., None]), -1).softmax(-1)
    expected = weights[..., :-1] @ v
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gates, weights[..., :-1].sum(-1))
    args = (*tensors, w, b)
    actual_grads = torch.autograd.grad(actual.square().sum() + gates.sum(), args, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum() + weights[..., :-1].sum(), args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_anti_collapse_gradient_and_supplied_formula():
    gates = torch.full((2, 4, 7), .02, requires_grad=True)
    loss, means = auxiliary_loss([gates])
    loss.backward()
    assert (gates.grad < 0).all()  # gradient descent increases physical mass
    supplied, _ = auxiliary_loss([gates], "supplied")
    torch.testing.assert_close(supplied, 4 * (means * -torch.log(means + 1e-6)).sum())


def test_token_blocks(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(17, dtype="<u4").tofile(path)
    data = TokenBlocks(path, 4, 20)
    assert len(data) == 4
    assert data[2].tolist() == [8, 9, 10, 11]
    with pytest.raises(ValueError):
        TokenBlocks(path, 4, 16)
    with pytest.raises(IndexError):
        data[4]


def test_bfloat16_checkpointed_attention_is_finite_and_causal():
    q, k, v = [torch.randn(1, 2, 5, 4, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3)]
    w = torch.randn(2, 4, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, gates = sink_attention(q, k, v, w, b, 2)
        loss = out[:, :, 2].float().square().sum() + gates[:, :, 2].sum()
    loss.backward()
    assert out.dtype == torch.bfloat16
    for tensor in (q, k, v, w, b):
        assert torch.isfinite(tensor.grad).all()
    assert k.grad[:, :, 3:].count_nonzero() == 0
    assert v.grad[:, :, 3:].count_nonzero() == 0


def tiny_qwen():
    pytest.importorskip("transformers", minversion="4.51.3")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32,
                         attention_dropout=0, use_sliding_window=False,
                         sliding_window=None, use_cache=False)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config)


def test_qwen_pretrained_preservation_gqa_rope_and_zero_sink_limit():
    from virtualsink.uptraining.qwen import adapt_qwen, disable_sinks
    model = tiny_qwen().eval()
    reference = copy.deepcopy(model)
    adapt_qwen(model, chunk_size=3, sink_bias=-1e6)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    tokens = torch.randint(0, 31, (2, 7))
    torch.testing.assert_close(model(tokens).logits, reference(tokens).logits, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.sink_b.fill_(3)
    disable_sinks(model)
    torch.testing.assert_close(model(tokens).logits, reference(tokens).logits, atol=1e-6, rtol=1e-5)
    disable_sinks(model, False)
    assert not torch.allclose(model(tokens).logits, reference(tokens).logits)
    with pytest.raises(ValueError, match="padding"):
        model(tokens, attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1]] * 2))


@pytest.mark.parametrize("mode", ["attention", "full"])
def test_checkpointed_qwen_loss_freezing_gradients_and_reload(mode):
    from virtualsink.uptraining.qwen import adapt_qwen, parameter_groups
    from virtualsink.uptraining.train import language_loss
    model = adapt_qwen(tiny_qwen(), chunk_size=3)
    parameter_groups(model, mode)
    tokens = torch.randint(0, 31, (2, 7))
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    loss, gates = language_loss(model, tokens, 3)
    native_loss = reference(tokens, labels=tokens).loss
    torch.testing.assert_close(loss, native_loss)
    penalty, _ = auxiliary_loss(gates)
    (loss + .01 * penalty).backward()
    ref_loss, ref_gates = language_loss(reference, tokens, 7)
    (ref_loss + .01 * auxiliary_loss(ref_gates)[0]).backward()
    for (name, p), (_, r) in zip(model.named_parameters(), reference.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, r.grad, atol=1e-6, rtol=1e-4)
        else:
            assert p.grad is None
    if mode == "attention":
        assert not model.model.embed_tokens.weight.requires_grad
        assert not model.model.layers[0].mlp.gate_proj.weight.requires_grad
    assert model.model.layers[0].self_attn.sink_w.grad.abs().sum() > 0
    restored = adapt_qwen(tiny_qwen(), chunk_size=3)
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(tokens).logits, restored(tokens).logits)
