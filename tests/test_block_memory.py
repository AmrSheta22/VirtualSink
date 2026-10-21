"""Independent arithmetic, causal streaming, and training checks for block memory."""
import copy
from dataclasses import replace

import pytest
import torch

from virtualsink.uptraining.block_memory import BlockMemoryLM
from virtualsink.uptraining.native import NativeLM, set_window
from virtualsink.uptraining.qwen import disable_sinks


def tiny(capacity=3, block_size=None):
    torch.manual_seed(42)
    config = dict(model_type="qwen2", vocab_size=23, hidden_size=16, intermediate_size=24,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  hidden_act="silu", rope_theta=10000., rms_norm_eps=1e-6,
                  tie_word_embeddings=True)
    model = BlockMemoryLM(NativeLM(config, 4), capacity, block_size)
    with torch.no_grad():
        for bank in model.memory:
            bank.role_bias.fill_(0.)
            bank.null_bias.fill_(-2.)
    return model


def state_close(a, b):
    assert (a.position, a.capacity, a.block_size) == (b.position, b.capacity, b.block_size)
    for left, right in zip(a.layers, b.layers):
        for x, y in zip(left.tensors(), right.tensors()):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("capacity,block", [(1, 1), (3, 3), (3, 2), (20, 20)])
def test_disabled_parity_and_all_stream_partitions(capacity, block):
    model = tiny(capacity, block)
    set_window(model.base, capacity)
    disable_sinks(model.base)
    tokens = torch.randint(0, 23, (2, 15))
    with torch.no_grad():
        disabled = model(tokens, read_memory=False, read_null=False)
        expected = model.base.model(tokens).last_hidden_state
        torch.testing.assert_close(disabled.hidden, expected, atol=2e-6, rtol=2e-5)
        full = model(tokens, diagnostics=True)
        for cuts in ([1] * 15, [2, 5, 1, 7], [7, 8]):
            state, outputs, start = None, [], 0
            for length in cuts:
                result = model(tokens[:, start:start + length], state)
                state, start = result.state, start + length
                outputs.append(result.hidden)
            torch.testing.assert_close(torch.cat(outputs, 1), full.hidden, atol=2e-6, rtol=2e-5)
            state_close(state, full.state)
        assert full.state.layers[0].memory_key.shape[2] == 3 * (max(0, 15 - capacity) // block)


def test_pooling_matches_independent_closed_forms_and_resets():
    model = tiny()
    bank = model.memory[0]
    saved = model.initial_state(2).layers[0]
    keys, values = torch.randn(2, 4, 3, 4), torch.randn(2, 4, 3, 4)
    importance = torch.rand(2, 4, 3)
    importance[0, 0] = 0  # explicit uniform fallback
    for i in range(3):
        saved = bank.accumulate(saved, keys[:, :, i], values[:, :, i], importance[:, :, i], i + 1, 3)
        assert saved.memory_key.shape[2] == (3 if i == 2 else 0)
    w = importance + 1e-8
    w = w / w.sum(-1, keepdim=True)
    free = ((keys * bank.pool_query[None, :, None]).sum(-1) / 2).softmax(-1)
    for actual, inputs in ((saved.memory_key, keys), (saved.memory_value, values)):
        expected = torch.stack((inputs.mean(2), (w[..., None] * inputs).sum(2),
                                (free[..., None] * inputs).sum(2)), 2)
        torch.testing.assert_close(actual, expected)
    assert saved.sum_key.count_nonzero() == 0
    assert saved.free_sum.count_nonzero() == 0
    assert torch.isneginf(saved.free_max).all()


def test_importance_is_received_attention_and_publication_has_exact_timing():
    model = tiny()
    layer, bank = model.base.model.layers[0], model.memory[0]
    with torch.no_grad():
        # All physical logits zero: without virtual reads each physical token
        # receives 1/(current window length), independent of its content.
        layer.self_attn.q_proj.weight.zero_()
        layer.self_attn.q_proj.bias.zero_()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        before = model(tokens[:, :5], read_memory=False, read_null=False)
        assert before.state.layers[0].memory_key.shape[2] == 0
        after = model(tokens[:, 5:], before.state, read_memory=False, read_null=False)
        assert after.state.layers[0].memory_key.shape[2] == 3
        inputs = model.base.model.embed_tokens(tokens[:, :3])
        z = layer.input_layernorm(inputs)
        v = layer.self_attn.v_proj(z).view(1, 3, 2, 4).transpose(1, 2).repeat_interleave(2, 1)
        received = torch.tensor([1 + 1/2 + 1/3, 1/2 + 1/3 + 1/3, 1.])
        expected = (v * (received / received.sum())[None, None, :, None]).sum(2)
        torch.testing.assert_close(after.state.layers[0].memory_value[:, :, 1], expected)
        torch.testing.assert_close(after.state.layers[0].importance,
                                   torch.tensor([1., 2/3, 1/3])[None, None].expand(1, 4, 3))


def test_causality_batch_reset_and_functional_state():
    model = tiny()
    tokens = torch.randint(0, 23, (2, 14))
    changed = tokens.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 23
    with torch.no_grad():
        full, altered = model(tokens), model(changed)
        torch.testing.assert_close(full.hidden[:, :7], altered.hidden[:, :7])
        torch.testing.assert_close(full.hidden[:1], model(tokens[:1]).hidden)
        torch.testing.assert_close(full.hidden, model(tokens).hidden)
        first = model(tokens[:, :8])
        snapshot = copy.deepcopy(first.state)
        model(tokens[:, 8:], first.state)
        state_close(first.state, snapshot)
        detached = full.state.detach()
        assert all(not x.requires_grad for s in detached.layers for x in s.tensors())
        for s in detached.layers:
            assert s.key.storage().size() == s.key.numel()


@pytest.mark.parametrize("checked", [False, True])
def test_delayed_gradients_and_optimizer_ownership(checked):
    model = tiny()
    groups = model.parameter_groups("memory")
    assert {id(p) for g in groups for p in g["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
    model.checkpointing = checked
    tokens = torch.randint(0, 23, (1, 11))
    first = model(tokens[:, :7])
    first.state.layers[0].memory_value.retain_grad()
    final = model(tokens[:, 7:], first.state)
    loss = torch.nn.functional.cross_entropy(model.base.lm_head(final.hidden[:, -1]), torch.tensor([3]))
    loss.backward()
    assert first.state.layers[0].memory_value.grad.abs().sum() > 0
    for name, p in model.memory.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_checkpoint_outputs_states_and_gradients_match():
    plain = tiny()
    plain.parameter_groups("attention")
    checked = copy.deepcopy(plain)
    checked.checkpointing = True
    tokens = torch.randint(0, 23, (2, 11))
    a, b = plain(tokens), checked(tokens)
    torch.testing.assert_close(a.hidden, b.hidden)
    state_close(a.state, b.state)
    a.hidden.square().sum().backward()
    b.hidden.square().sum().backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_value_and_role_interventions_and_checkpoint_validation():
    model = tiny()
    tokens = torch.randint(0, 23, (1, 12))
    with torch.no_grad():
        full = model(tokens, diagnostics=True)
        zero = model(tokens, zero_memory_values=True)
        disabled = model(tokens, role_mask=(False, False, False), diagnostics=True)
        assert (full.hidden[:, 6:] - zero.hidden[:, 6:]).abs().max() > 1e-5
        assert disabled.diagnostics[..., 1:].count_nonzero() == 0
        assert full.diagnostics[:, :, :, :5, 1:].count_nonzero() == 0
    target = tiny()
    target.load_state_dict(model.state_dict())
    torch.testing.assert_close(target(tokens).hidden, full.hidden)
    with pytest.raises(ValueError, match="checkpoint"):
        tiny(block_size=2).load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="metadata"):
        model(tokens, replace(full.state, block_size=2))


def test_extreme_free_pooling_stays_finite():
    model = tiny()
    bank = model.memory[0]
    with torch.no_grad():
        bank.pool_query.fill_(1000.)
    saved = model.initial_state(1).layers[0]
    keys = torch.tensor([1000., -1000., 999.])[None, None, :, None].expand(1, 4, 3, 4)
    values = torch.randn_like(keys)
    for i in range(3):
        saved = bank.accumulate(saved, keys[:, :, i], values[:, :, i], torch.ones(1, 4), i + 1, 3)
    assert torch.isfinite(saved.memory_value).all()
    torch.testing.assert_close(saved.memory_value[:, :, 2], values[:, :, 0])


def test_pilot_evaluator_scores_full_articles_and_counts_regions(monkeypatch):
    import io
    import json
    from hpc.run_block_memory import evaluate
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    model = tiny(128)

    class Data:
        def length(self, index):
            return 270

        def segment(self, index, start, count):
            ids = torch.arange(start, min(start + count + 1, 270))[None] % 23
            return ids[:, :-1], ids[:, 1:]

    output = io.StringIO()
    report = evaluate(model, Data(), [0], 'test', {}, output)
    assert report['all']['tokens'] == 269
    assert report['post_publication']['tokens'] == 14
    assert report['4-8K']['ce'] is None
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert sum(row['tokens'] for row in rows) == 269
    assert rows[-1]['start'] == 256
