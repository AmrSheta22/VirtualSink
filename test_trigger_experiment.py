"""Dataset, ablation, metrics, and Slurm-only end-to-end regression tests."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from run_ablation import MSEAccumulator, load_checkpoint
from synthetic_dataset import TriggerConditionalDataset
from toy_models import ATTENTION_TYPES, ToyTransformer


def test_exact_dataset_definition():
    data = TriggerConditionalDataset(100, seed=31)
    observed = set()
    for sample in data:
        x, y, j = sample["x"], sample["y"], int(sample["trigger_index"])
        observed.add(j)
        assert x.shape == y.shape == (64, 64)
        assert 2 <= j <= 62
        assert x[0, 0] == 1 and x[:, 0].sum() == 1
        assert x[j, 1] == 1 and x[:, 1].sum() == 1
        assert x[0, 2] == x[j, 2] == 0 and x[:, 2].sum() == 62
        assert sample["trigger_mask"].sum() == 1
        torch.testing.assert_close(y[j], x[1:j].mean(0), rtol=0, atol=0)
        assert torch.count_nonzero(y[~sample["trigger_mask"]]) == 0
        assert torch.count_nonzero(x[:, 3:]) == 64 * 61
    assert len(observed) > 30


def test_data_deterministic_and_rng_isolated():
    a, b = TriggerConditionalDataset(5, seed=3), TriggerConditionalDataset(5, seed=4)
    state = torch.random.get_rng_state()
    first = a[2]
    _ = a[0]
    for key in first:
        assert torch.equal(first[key], a[2][key])
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.equal(a[2]["x"], b[2]["x"])


@pytest.mark.parametrize("kwargs", [{"seq_len": 3}, {"d": 3}, {"num_samples": 0}])
def test_invalid_dataset(kwargs):
    with pytest.raises(ValueError):
        TriggerConditionalDataset(**kwargs)


@pytest.mark.parametrize("attn_type", ATTENTION_TYPES)
def test_attention_denominator_formula(attn_type):
    torch.manual_seed(9)
    attn = ATTENTION_TYPES[attn_type](4, 1)
    with torch.no_grad():
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            projection.weight.copy_(torch.eye(4))
    x = torch.randn(1, 4, 4) * 0.1
    actual, info = attn(x, return_attention=True)
    scores = (x @ x.transpose(-1, -2)) / 2
    scores = scores.masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), -torch.inf)
    numerator = scores.exp()  # Small reference-only logits; production uses stable softmax.
    denominator = numerator.sum(-1, keepdim=True)
    if attn_type == "quiet":
        denominator = denominator + 1
    elif attn_type == "static_sink":
        denominator = denominator + attn.sink_b.exp().reshape(1, 1, 1)
    elif attn_type == "query_virtual_sink":
        s = (x * attn.sink_w[0]).sum(-1) + attn.sink_b[0]
        denominator = denominator + s.exp()[..., None]
    weights = numerator / denominator
    torch.testing.assert_close(actual, weights @ x)
    torch.testing.assert_close(info["physical_weights"][:, 0], weights)
    torch.testing.assert_close(info["physical_weights"].sum(-1) + info["sink_probability"],
                               torch.ones(1, 1, 4))
    assert not info["physical_weights"].requires_grad
    torch.testing.assert_close(attn(x), actual)


def test_only_query_sink_depends_on_query():
    query = torch.randn(2, 2, 4, 4)
    static = ATTENTION_TYPES["static_sink"](8, 2)
    torch.testing.assert_close(static._sink_logits(query), static._sink_logits(query + 2))
    conditioned = ATTENTION_TYPES["query_virtual_sink"](8, 2)
    with torch.no_grad():
        conditioned.sink_w.fill_(1)
    torch.testing.assert_close(conditioned._sink_logits(query + 2) - conditioned._sink_logits(query),
                               torch.full((2, 2, 4), 8.0))


def test_shared_initialization_is_identical():
    models = []
    for name in ATTENTION_TYPES:
        torch.manual_seed(123)
        models.append(ToyTransformer(name, d_model=8, n_heads=2))
    base = models[0].state_dict()
    for model in models[1:]:
        for key, tensor in base.items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
    assert not any("sink" in name for name, _ in models[0].named_parameters())
    assert not any("sink" in name for name, _ in models[1].named_parameters())
    assert all("sink_w" not in name for name, _ in models[2].named_parameters())


@pytest.mark.parametrize("attn_type", ATTENTION_TYPES)
@pytest.mark.parametrize("layers", [1, 2])
def test_transformer_shape_causality_and_gradients(attn_type, layers):
    torch.manual_seed(4)
    model = ToyTransformer(attn_type, d_model=8, n_heads=2, num_layers=layers)
    x = torch.randn(2, 6, 8, requires_grad=True)
    y, diagnostics = model(x, return_attention=True)
    assert y.shape == x.shape and len(diagnostics) == layers
    y[:, 3].square().sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if attn_type == "query_virtual_sink":
        assert model.blocks[0].attn.sink_w.grad.abs().sum() > 0


def test_metric_weighting_and_zero_predictor():
    data = TriggerConditionalDataset(3, d=8, seq_len=8)
    metrics = MSEAccumulator()
    for sample in data:
        metrics.update(torch.zeros_like(sample["y"]), sample["y"], sample["trigger_mask"])
    result = metrics.compute()
    assert result["non_trigger_mse"] == 0
    assert result["trigger_mse"] > 0
    assert result["mse"] == pytest.approx(result["trigger_mse"] / 8)


@pytest.mark.skipif(not os.environ.get("SLURM_JOB_ID"), reason="Repository requires training through Slurm")
def test_end_to_end_training_and_plotting(tmp_path):
    output = tmp_path / "smoke"
    subprocess.run([sys.executable, "run_ablation.py", "--output-dir", str(output),
                    "--device", "cpu", "--steps", "2", "--batch-size", "2",
                    "--d-model", "8", "--seq-len", "8", "--n-heads", "2",
                    "--validation-samples", "4", "--test-samples", "4",
                    "--eval-every", "1", "--threads", "1"],
                   cwd=Path(__file__).parent, check=True, timeout=120)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["variants"]) == 4
    assert (output / "trigger_experiment_results.png").stat().st_size > 1000
    assert (output / "convergence.png").stat().st_size > 1000
    for name in ATTENTION_TYPES:
        checkpoint = load_checkpoint(output / name / "best.pt")
        assert checkpoint["step"] in (1, 2)
        assert checkpoint["model_config"]["attn_type"] == name
    report = json.loads((output / "mechanism_summary.json").read_text())
    assert len(report["statistics"]) == 4
