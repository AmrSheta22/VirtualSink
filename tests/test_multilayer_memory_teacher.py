import copy
import pytest
import torch
from test_recurrent_sink import tiny, assert_state_close


def test_teacher_matches_dominant_scores_on_its_own_multilayer_trajectory():
    model = tiny()
    reference = copy.deepcopy(model)
    for bank in reference.memory:
        original = bank.read
        def read(*args, original=original):
            null, scores, values = original(*args)
            occupied = args[-1].abs().sum((1, 2, 3)) > 0
            return null, scores + occupied[:, None, None] * 100., values
        bank.read = read
    tokens = torch.randint(0, 23, (2, 11))
    teacher = model(tokens, memory_teacher=True, diagnostics=True)
    actual = reference(tokens)
    torch.testing.assert_close(teacher.hidden, actual.hidden, atol=2e-6, rtol=2e-5)
    assert_state_close(teacher.state, actual.state)
    assert (teacher.diagnostics['memory_mass'][..., 3:] - 1).abs().max() < 1e-6
    ordinary = model(tokens)
    torch.testing.assert_close(teacher.hidden[:, :3], ordinary.hidden[:, :3])
    assert not torch.allclose(teacher.hidden[:, 3:], ordinary.hidden[:, 3:])


def test_teacher_checkpoint_and_connected_segment_state():
    model = tiny()
    reference = copy.deepcopy(model)
    model.checkpointing = True
    tokens = torch.randint(0, 23, (2, 11))
    a = model(tokens[:, :5], memory_teacher=True)
    b = model(tokens[:, 5:], a.state, memory_teacher=True)
    whole = reference(tokens, memory_teacher=True)
    joined = torch.cat((a.hidden, b.hidden), 1)
    torch.testing.assert_close(joined, whole.hidden, atol=2e-6, rtol=2e-5)
    joined.square().mean().backward()
    whole.hidden.square().mean().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        model(tokens, memory_teacher=True, read_memory=False)
