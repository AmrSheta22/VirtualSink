"""Behavioral and numerical checks for virtual sink attention (CPU compatible)."""

import pytest
import torch
from torch.nn import functional as F

from query_virtual_sink_attention import QueryVirtualSinkAttention


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1234)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [None, 1, 3])
def test_shape_dtype_and_gradients(dtype, window):
    model = QueryVirtualSinkAttention(16, 4, window_size=window, dtype=dtype)
    x = torch.randn(2, 6, 16, dtype=dtype, requires_grad=True)
    output = model(x)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().sum().backward()
    for tensor in (x, model.sink_w, model.sink_b, model.q_proj.weight,
                   model.k_proj.weight, model.v_proj.weight, model.out_proj.weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_initialization():
    model = QueryVirtualSinkAttention(512, 8)
    assert model.sink_w.shape == (8, 64)
    assert model.sink_b.shape == (8,)
    assert torch.equal(model.sink_b, torch.full((8,), -3.5))
    assert abs(model.sink_w.mean().item()) < 2e-5
    assert 0.8e-4 < model.sink_w.std().item() < 1.2e-4
    assert QueryVirtualSinkAttention(8, 2, sink_bias_init=-5).sink_b.tolist() == [-5, -5]


@pytest.mark.parametrize("window", [None, 1, 3, 20])
def test_lower_bound_matches_sdpa(window):
    model = QueryVirtualSinkAttention(16, 4, window_size=window)
    with torch.no_grad():
        model.sink_b.fill_(-1e6)
    x = torch.randn(2, 7, 16)
    q, k, v = [proj(x).reshape(2, 7, 4, 4).transpose(1, 2)
               for proj in (model.q_proj, model.k_proj, model.v_proj)]
    i = torch.arange(7)
    mask = i[None, :] <= i[:, None]
    if window is not None:
        mask &= i[None, :] > i[:, None] - window
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref = model.out_proj(ref.transpose(1, 2).reshape(2, 7, 16))
    assert (model(x) - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_logits_are_finite_in_forward_and_backward(dtype, sign):
    model = QueryVirtualSinkAttention(8, 2, dtype=dtype)
    # 1e6 is not representable by FP16 parameters; 1e4 still saturates softmax.
    magnitude = 1e4 if dtype == torch.float16 else 1e6
    with torch.no_grad():
        model.sink_b.fill_(sign * magnitude)
    x = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    y = model(x)
    assert torch.isfinite(y).all()
    if sign > 0:
        assert y.float().norm().item() < 1e-5
    y.float().sum().backward()
    for tensor in (x, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_explicit_zero_value_reference():
    model = QueryVirtualSinkAttention(12, 3)
    x = torch.randn(2, 5, 12)
    with torch.no_grad():
        model.sink_w.normal_(0, 0.5)
        model.sink_b.normal_()
    q, k, v = [proj(x).reshape(2, 5, 3, 4).transpose(1, 2)
               for proj in (model.q_proj, model.k_proj, model.v_proj)]
    scores = q @ k.transpose(-1, -2) / 2
    scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), -torch.inf)
    sink = (q * model.sink_w[None, :, None, :]).sum(-1) + model.sink_b[None, :, None]
    weights = torch.cat([scores, sink[..., None]], -1).softmax(-1)
    zero_extended_v = torch.cat([v, torch.zeros(2, 3, 1, 4)], -2)
    ref = model.out_proj((weights @ zero_extended_v).transpose(1, 2).reshape(2, 5, 12))
    torch.testing.assert_close(model(x), ref)
    assert (weights[..., :-1].sum(-1) < 1).all()


def test_sliding_window_and_causal_dependencies():
    model = QueryVirtualSinkAttention(8, 2, window_size=3)
    x = torch.randn(1, 8, 8, requires_grad=True)
    model(x)[:, 5].square().sum().backward()
    assert torch.count_nonzero(x.grad[:, :3]) == 0  # t-w and earlier
    assert torch.count_nonzero(x.grad[:, 6:]) == 0  # future
    assert (x.grad[:, 3:6].abs().sum(-1) > 0).all()


def test_window_override_and_separate_inputs():
    model = QueryVirtualSinkAttention(10, 3, d_head=4, window_size=2)
    x = torch.randn(2, 6, 10)
    torch.testing.assert_close(model(x), model(x, x, x))
    torch.testing.assert_close(model(x), model(q=x, k=x, v=x))
    full = model(x, sliding_window=4)
    tail = model(q=x[:, -2:], k=x, v=x, sliding_window=4)
    torch.testing.assert_close(tail, full[:, -2:])
    assert not torch.allclose(full, model(x))


def test_boolean_additive_and_fully_masked_rows():
    model = QueryVirtualSinkAttention(8, 2)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)
    mask[..., 1] = False
    mask[:, :, 3, :] = False
    additive = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    y = model(x, attention_mask=mask)
    torch.testing.assert_close(y, model(x, attention_mask=additive))
    assert torch.count_nonzero(y[:, 3]) == 0
    y[:, 4].sum().backward()
    assert torch.count_nonzero(x.grad[:, 1]) == 0
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_cpu_autocast():
    model = QueryVirtualSinkAttention(8, 2)
    x = torch.randn(2, 5, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(x)
    assert y.dtype == torch.bfloat16
    y.float().square().sum().backward()
    assert torch.isfinite(model.sink_w.grad).all()


def test_double_precision_gradcheck():
    model = QueryVirtualSinkAttention(4, 2).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    def evaluate(x, w, b):
        return torch.func.functional_call(model, {"sink_w": w, "sink_b": b}, (x,))
    assert torch.autograd.gradcheck(evaluate, (x, model.sink_w, model.sink_b))


@pytest.mark.parametrize("kwargs", [
    {"d_model": 0}, {"n_heads": 0}, {"d_model": 7}, {"d_head": 0},
    {"window_size": 0}, {"window_size": True}, {"dtype": torch.int64},
    {"sink_bias_init": float("nan")},
])
def test_invalid_constructor(kwargs):
    options = {"d_model": 8, "n_heads": 2, **kwargs}
    with pytest.raises(ValueError):
        QueryVirtualSinkAttention(**options)


def test_invalid_forward():
    model = QueryVirtualSinkAttention(8, 2)
    x = torch.randn(2, 4, 8)
    for call in (lambda: model(), lambda: model(x, q=x), lambda: model(x, k=x),
                 lambda: model(x[..., :3]), lambda: model(x, x[:, :2], x[:, :2]),
                 lambda: model(x, sliding_window=-1),
                 lambda: model(x, attention_mask=torch.ones(4, 4, dtype=torch.int64))):
        with pytest.raises(ValueError):
            call()
