import pytest
import torch

from virtualsink.uptraining.recall_experiment import examples, make_model, initialize_mass
from virtualsink.uptraining.memory_warmup import ARMS, auxiliary_logits, optimizer_for_phase


def test_direct_loss_reaches_memory_without_updating_backbone_or_scores():
    model = make_model(1729, -4., "cpu")
    initialize_mass(model, 1729, 1e-4, "cpu")
    optimizer = optimizer_for_phase(model, ARMS[2])
    saved = {name: p.detach().clone() for name, p in model.base.named_parameters()}
    tokens, target = examples(8, 24, torch.Generator().manual_seed(27), "cpu")
    result = model(tokens)
    joint = torch.nn.functional.cross_entropy(model.base.lm_head(result.hidden[:, -1]), target)
    joint.backward(retain_graph=True)
    weak = model.memory[0].write_weight.grad.norm().item()
    optimizer.zero_grad(set_to_none=True)
    direct = torch.nn.functional.cross_entropy(auxiliary_logits(model, tokens, result.state), target)
    direct.backward()
    strong = model.memory[0].write_weight.grad.norm().item()
    assert strong > 100 * weak
    assert all(p.grad is None for p in model.base.parameters())
    assert model.memory[0].memory_bias.grad is None
    assert model.memory[0].read_key.grad is None
    optimizer.step()
    for name, parameter in model.base.named_parameters():
        torch.testing.assert_close(parameter, saved[name], atol=0, rtol=0)
    transferred = optimizer_for_phase(model, ARMS[2], adapting=True, previous=optimizer)
    weight = model.memory[0].write_weight
    assert transferred.state[weight] is optimizer.state[weight]
    assert model.memory[0].read_key.requires_grad
    assert model.base.model.layers[0].self_attn.q_proj.weight.requires_grad


def test_auxiliary_decoder_needs_stored_content_and_has_no_new_parameters():
    model = make_model(1729, -4., "cpu")
    names = list(dict(model.named_parameters()))
    tokens, _ = examples(4, 24, torch.Generator().manual_seed(28), "cpu")
    result = model(tokens)
    logits = auxiliary_logits(model, tokens, result.state)
    assert logits.shape == (4, 32) and torch.isfinite(logits).all()
    assert logits.abs().sum() > 0
    empty = model(tokens, write_memory=False)
    erased = auxiliary_logits(model, tokens, empty.state)
    assert erased.count_nonzero() == 0
    assert list(dict(model.named_parameters())) == names
