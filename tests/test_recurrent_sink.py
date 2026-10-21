"""No-download numerical and state-lifecycle tests for recurrent virtual memory."""

import copy

import pytest
import torch

from virtualsink.uptraining.native import NativeLM, set_window
from virtualsink.uptraining.qwen import disable_sinks
from virtualsink.uptraining.recurrent_sink import (
    LayerMemoryState, MemoryBank, MemoryState, RecurrentSinkLM,
)


def tiny(capacity=3):
    torch.manual_seed(42)
    config = dict(model_type="qwen2", vocab_size=23, hidden_size=16, intermediate_size=24,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  hidden_act="silu", rope_theta=10000., rms_norm_eps=1e-6,
                  tie_word_embeddings=True)
    return RecurrentSinkLM(NativeLM(config, 4), capacity, (4., 32., 128.))


def assert_state_close(a, b):
    assert (a.position, a.capacity) == (b.position, b.capacity)
    for actual, expected in zip(a.layers, b.layers):
        for name in ("key", "value", "cell", "exposed"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("capacity", [1, 3, 20])
def test_disabled_virtuals_match_native_sliding_attention(capacity):
    model = tiny(capacity)
    set_window(model.base, capacity)
    disable_sinks(model.base)
    tokens = torch.randint(0, 23, (2, 11))
    with torch.no_grad():
        actual = model(tokens, read_memory=False, read_null=False, diagnostics=True)
        expected = model.base.model(tokens).last_hidden_state
    torch.testing.assert_close(actual.hidden, expected, atol=2e-6, rtol=2e-5)
    assert actual.diagnostics["memory_mass"].count_nonzero() == 0
    assert actual.diagnostics["null_mass"].count_nonzero() == 0


def test_full_segment_and_tokenwise_parity_with_appreciable_memory():
    model = tiny()
    for bank in model.memory:
        with torch.no_grad():
            bank.memory_bias.fill_(0.)
            bank.null_bias.fill_(-2.)
    tokens = torch.randint(0, 23, (2, 13))
    with torch.no_grad():
        full = model(tokens, diagnostics=True)
        first = model(tokens[:, :5])
        snapshot = copy.deepcopy(first.state)
        second = model(tokens[:, 5:], first.state)
        torch.testing.assert_close(full.hidden, torch.cat((first.hidden, second.hidden), 1))
        assert_state_close(first.state, snapshot)  # no caller-owned tensor was modified
        assert_state_close(full.state, second.state)
        state, logits = None, []
        for token in tokens.unbind(1):
            output, state, _ = model.step(token, state)
            logits.append(output)
        torch.testing.assert_close(model.base.lm_head(full.hidden), torch.stack(logits, 1), atol=1e-5, rtol=2e-5)
        assert_state_close(full.state, state)
    assert full.diagnostics["memory_mass"].mean() > .05
    for saved in full.state.layers:
        assert saved.key.shape[2] == model.capacity
        assert saved.cell.abs().sum() > 0


def test_causality_batch_isolation_reset_and_no_write_before_eviction():
    model = tiny()
    tokens = torch.randint(0, 23, (2, 9))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 23
    with torch.no_grad():
        full = model(tokens)
        altered = model(changed)
        torch.testing.assert_close(full.hidden[:, :5], altered.hidden[:, :5])
        one = model(tokens[:1])
        torch.testing.assert_close(full.hidden[:1], one.hidden)
        repeated = model(tokens)
        torch.testing.assert_close(full.hidden, repeated.hidden)
        before = model(tokens[:, :3])
        after = model(tokens[:, 3:4], before.state)
        no_write = model(tokens, write_memory=False)
    assert all(s.cell.count_nonzero() == 0 for s in before.state.layers)
    assert all(s.cell.count_nonzero() > 0 for s in after.state.layers)
    assert all(s.cell.count_nonzero() == 0 for s in no_write.state.layers)


def test_lstm_retention_and_forget_ablation_match_closed_form():
    bank = MemoryBank(1, 2, (4., 32., 128.))
    with torch.no_grad():
        bank.write_weight.zero_()
        bank.recurrent_weight.zero_()
        # Candidate zero means the input gate introduces no new content.
    cell = torch.ones(1, 1, 3, 2)
    exposed = torch.zeros_like(cell)
    key = value = torch.zeros(1, 1, 2)
    for _ in range(10):
        cell, exposed = bank.write(key, value, cell, exposed)
    expected = torch.exp(-10 / bank.timescales)[None, None, :, None].expand_as(cell)
    torch.testing.assert_close(cell, expected, atol=2e-6, rtol=2e-5)
    kept, _ = bank.write(key, value, cell, exposed, forget_override=1.)
    erased, _ = bank.write(key, value, cell, exposed, forget_override=0.)
    torch.testing.assert_close(kept, cell)
    assert erased.count_nonzero() == 0


@pytest.mark.parametrize("checkpointing", [False, True])
def test_delayed_loss_reaches_memory_and_optimizer_owns_every_parameter(checkpointing):
    model = tiny()
    groups = model.parameter_groups("memory")
    owned = {id(p) for g in groups for p in g["params"]}
    assert owned == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in model.base.parameters())
    for bank in model.memory:
        with torch.no_grad():
            bank.memory_bias.fill_(0.)
    model.checkpointing = checkpointing
    tokens = torch.randint(0, 23, (1, 12))
    first = model(tokens[:, :6])
    first.state.layers[0].cell.retain_grad()
    final = model(tokens[:, 6:], first.state)
    loss = torch.nn.functional.cross_entropy(model.base.lm_head(final.hidden[:, -1]), torch.tensor([3]))
    loss.backward()
    assert first.state.layers[0].cell.grad.abs().sum() > 0
    for name, param in model.memory.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_checkpoint_recomputation_preserves_outputs_states_and_parameter_gradients():
    plain = tiny()
    plain.parameter_groups("attention")
    for bank in plain.memory:
        with torch.no_grad():
            bank.memory_bias.fill_(-1.)
    checked = copy.deepcopy(plain)
    checked.checkpointing = True
    tokens = torch.randint(0, 23, (2, 9))
    a, b = plain(tokens), checked(tokens)
    torch.testing.assert_close(a.hidden, b.hidden)
    assert_state_close(a.state, b.state)
    a.hidden.square().sum().backward()
    b.hidden.square().sum().backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_detach_cuts_history_but_preserves_memory_and_bounded_storage():
    model = tiny()
    tokens = torch.randint(0, 23, (1, 12))
    result = model(tokens)
    detached = result.state.detach()
    assert_state_close(result.state, detached)
    for layer in detached.layers:
        for tensor in (layer.key, layer.value, layer.cell, layer.exposed):
            assert tensor.grad_fn is None and not tensor.requires_grad
        assert layer.key.storage().size() == layer.key.numel()


def test_train_only_mass_calibration_is_measured_and_restores_training_mode():
    model = tiny()
    documents = [torch.randint(0, 23, (2, 15)), torch.randint(0, 23, (1, 12))]
    before = model.memory[0].write_weight.detach().clone()
    report = model.calibrate(documents, target_mass=1e-4, passes=3)
    assert model.training
    measured = torch.tensor(report["quantile_mass"])
    torch.testing.assert_close(measured, torch.full_like(measured, 1e-4), atol=2e-7, rtol=.005)
    torch.testing.assert_close(before, model.memory[0].write_weight)
    with torch.no_grad():
        adapted = model(documents[0]).hidden
        control = model(documents[0], read_null=False, read_memory=False).hidden
    assert (adapted - control).abs().max() < .01


def test_serialization_and_invalid_states():
    model = tiny()
    other = tiny()
    other.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="mismatch"):
        tiny(4).load_state_dict(model.state_dict())
    tokens = torch.ones(1, 2, dtype=torch.long)
    state = model(tokens).state
    wrong = MemoryState(state.layers, state.position, 99)
    with pytest.raises(ValueError, match="capacity"):
        model(tokens, wrong)
    with pytest.raises(ValueError, match="shape"):
        model(tokens.expand(2, -1), state)
    with pytest.raises(ValueError, match="override"):
        model(tokens, forget_override=float("nan"))
    with pytest.raises(ValueError, match="longer"):
        model.calibrate([tokens])


def test_calibration_bypasses_disruptive_provisional_biases():
    model = tiny()
    for bank in model.memory:
        with torch.no_grad():
            bank.null_bias.fill_(100.)
            bank.memory_bias.fill_(100.)
    tokens = torch.randint(0, 23, (2, 17))
    with torch.no_grad():
        prospective = model(tokens, diagnostics=True, bypass_virtual=True)
        control = model(tokens, read_memory=False, read_null=False)
        torch.testing.assert_close(prospective.hidden, control.hidden)
        assert prospective.diagnostics["memory_mass"].count_nonzero() == 0
        assert torch.isfinite(prospective.diagnostics["virtual_log_partition"]).all()
    report = model.calibrate([tokens])
    assert report["quantile"] == 1.
    assert max(max(row) for row in report["max_mass"]) < 1.01e-4
