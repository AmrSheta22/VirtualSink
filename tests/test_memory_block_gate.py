import copy
import torch

from virtualsink.uptraining.recall_experiment import examples, make_model, initialize_mass
from virtualsink.uptraining.memory_block_gate import full_block_logits, gate_loss, optimizer_for_phase, ARMS


def test_full_block_teacher_matches_actual_forward_with_memory_dominating():
    model = make_model(1729, -4., "cpu")
    tokens, _ = examples(4, 24, torch.Generator().manual_seed(28), "cpu")
    output = model(tokens)
    teacher = full_block_logits(model, tokens, output.state)
    bank = model.memory[0]
    read = bank.read
    def forced(*args):
        null, scores, values = read(*args)
        return null, scores + 100., values
    bank.read = forced
    actual = model.base.lm_head(model(tokens).hidden[:, -1])
    bank.read = read
    torch.testing.assert_close(teacher, actual, atol=2e-6, rtol=2e-5)


def test_connected_gate_log_odds_and_frozen_warmup_gradients():
    model = make_model(1729, -4., "cpu")
    initialize_mass(model, 1729, 1e-4, "cpu")
    optimizer = optimizer_for_phase(model, ARMS[3])
    tokens, target = examples(4, 24, torch.Generator().manual_seed(29), "cpu")
    output = model(tokens, diagnostics=True)
    info = output.diagnostics
    torch.testing.assert_close(info["memory_gate_logit"].sigmoid(), info["memory_mass"], atol=1e-8, rtol=2e-5)
    gate_loss(info).backward(retain_graph=True)
    assert model.memory[0].memory_bias.grad.norm() > .01
    assert model.memory[0].read_key.grad.norm() > 0
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(full_block_logits(model, tokens, output.state), target).backward()
    assert model.memory[0].write_weight.grad.norm() > 0
    assert all(p.grad is None for p in model.base.parameters())
    old = {name: p.detach().clone() for name, p in model.base.named_parameters()}
    optimizer.step()
    assert all(torch.equal(p, old[name]) for name, p in model.base.named_parameters())
    next_optimizer = optimizer_for_phase(model, ARMS[3], adapting=True, previous=optimizer)
    assert next_optimizer.state[model.memory[0].write_weight] is optimizer.state[model.memory[0].write_weight]


def test_gate_loss_matches_checkpoint_recomputation():
    plain = make_model(1730, -4., "cpu")
    checked = copy.deepcopy(plain)
    checked.checkpointing = True
    tokens, _ = examples(4, 24, torch.Generator().manual_seed(30), "cpu")
    for model in (plain, checked):
        model.train()
        optimizer_for_phase(model, ARMS[3])
        gate_loss(model(tokens, diagnostics=True).diagnostics).backward()
    for a, b in zip(plain.memory.parameters(), checked.memory.parameters()):
        if a.grad is None:
            assert b.grad is None or b.grad.count_nonzero() == 0
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
