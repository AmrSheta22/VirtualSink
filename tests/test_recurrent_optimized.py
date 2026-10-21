import copy
import pytest
import torch
from test_recurrent_sink import tiny, assert_state_close
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM


@pytest.mark.parametrize('capacity', [1, 3, 20])
@pytest.mark.parametrize('flags', [{}, {'write_memory': False}, {'read_memory': False},
    {'read_null': False}, {'zero_memory_values': True}, {'forget_override': 0.},
    {'bypass_virtual': True}, {'memory_teacher': True}])
def test_outputs_state_and_gradients(capacity, flags):
    reference = tiny(capacity)
    for bank in reference.memory:
        with torch.no_grad():
            bank.memory_bias.fill_(0.)
            bank.null_bias.fill_(-2.)
    optimized = OptimizedRecurrentSinkLM(copy.deepcopy(reference.base), capacity, (4., 32., 128.))
    optimized.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 23, (2, 13))
    outputs = []
    for model in (reference, optimized):
        model.parameter_groups('full')
        model.checkpointing = True
        first = model(tokens[:, :5], diagnostics=True, **flags)
        second = model(tokens[:, 5:], first.state, diagnostics=True, **flags)
        loss = first.hidden.square().mean() + second.hidden.square().mean()
        if flags.get('read_memory', True):
            loss = loss + second.diagnostics['memory_gate_logit'].mean()*.01
        loss.backward()
        outputs.append((first, second))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual.hidden, expected.hidden, atol=3e-6, rtol=3e-5)
        assert_state_close(actual.state, expected.state)
        for name in actual.diagnostics:
            torch.testing.assert_close(actual.diagnostics[name], expected.diagnostics[name], atol=3e-6, rtol=3e-5)
    for (name, a), (_, b) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)
