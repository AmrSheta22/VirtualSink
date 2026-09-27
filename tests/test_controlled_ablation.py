"""Checks that the controlled study isolates architecture and loss weighting."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from virtualsink.training.metrics import MSEAccumulator, regression_loss
from virtualsink.analysis.study import describe
from virtualsink.attention import ATTENTION_TYPES
from virtualsink.models import AttentionOnlyModel, ToyTransformer, build_model


@pytest.mark.parametrize("name", ATTENTION_TYPES)
def test_attention_only_has_no_bypass(name):
    model = AttentionOnlyModel(name, d_model=8, n_heads=2)
    x = torch.randn(2, 6, 8)
    torch.testing.assert_close(model(x), model.attn(x))
    prediction, info = model(x, return_attention=True)
    assert len(info) == 1 and prediction.shape == x.shape
    assert set(model._modules) == {"attn"}
    with torch.no_grad():
        model.attn.out_proj.weight.zero_()
    assert torch.count_nonzero(model(x)) == 0


def test_initialization_and_checkpoint_factory():
    states = []
    for name in ATTENTION_TYPES:
        torch.manual_seed(1)
        model = AttentionOnlyModel(name, d_model=8, n_heads=2)
        states.append(model.state_dict())
        copy = build_model(model.config)
        copy.load_state_dict(model.state_dict())
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(model(x), copy(x))
    for state in states[1:]:
        for key, value in states[0].items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    old = ToyTransformer(d_model=8, n_heads=2)
    assert isinstance(build_model(old.config), ToyTransformer)


def test_balanced_loss_equal_group_weight_and_gradient():
    target = torch.zeros(1, 4, 4)
    prediction = torch.tensor([[[2.] * 4, [1.] * 4, [1.] * 4, [1.] * 4]], requires_grad=True)
    mask = torch.tensor([[True, False, False, False]])
    assert regression_loss(prediction, target, mask).item() == pytest.approx(7 / 4)
    balanced = regression_loss(prediction, target, mask, "balanced")
    assert balanced.item() == pytest.approx(2.5)
    balanced.backward()
    assert prediction.grad[0, 0].sum().item() == pytest.approx(2.0)
    assert prediction.grad[0, 1:].sum().item() == pytest.approx(1.0)


def test_coordinate_and_batch_weighted_metrics():
    prediction = torch.zeros(3, 4, 8)
    target = torch.zeros_like(prediction)
    target[:, 1, 2] = 1
    target[:, 1, 3:] = 2
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[:, 1] = True
    one, split = MSEAccumulator(), MSEAccumulator()
    one.update(prediction, target, mask)
    for batch in (slice(0, 1), slice(1, 3)):
        split.update(prediction[batch], target[batch], mask[batch])
    result = one.compute()
    assert result == pytest.approx(split.compute())
    assert result["gaussian_trigger_mse"] == 4
    assert result["indicator_trigger_mse"] == pytest.approx(1 / 3)
    assert result["trigger_mse"] == pytest.approx(21 / 8)
    assert result["gaussian_non_trigger_mse"] == 0
    assert result["gaussian_balanced_mse"] == 2
    assert result["gaussian_mse"] == 1


def test_sample_standard_deviation():
    result = describe([1, 2, 3])
    assert result["mean"] == 2 and result["std"] == 1
    assert describe([1])["std"] is None


def test_controlled_cli_end_to_end(tmp_path):
    # Two-step CPU smoke test of module entry points, checkpoints and reporting.
    root = tmp_path / "control"
    subprocess.run([sys.executable, "-m", "virtualsink.experiments.controlled", "--output-dir", str(root),
                    "--device", "cpu", "--seeds", "5", "--steps", "2", "--batch-size", "2",
                    "--d-model", "8", "--seq-len", "8", "--n-heads", "2", "--threads", "1",
                    "--validation-samples", "4", "--test-samples", "4", "--eval-every", "1"],
                   cwd=tmp_path, check=True, timeout=120)
    report = json.loads((root / "study_summary.json").read_text())
    assert len(report["runs"]) == 2
    assert set(report["conditions"]) == {"token_mean", "balanced"}
    for entry in report["runs"]:
        for variant in entry["summary"]["variants"]:
            expected = "mse" if entry["loss_mode"] == "token_mean" else "balanced_mse"
            assert variant["selection_metric"] == expected
    assert (root / "controlled_comparison.png").stat().st_size > 1000
    assert (root / "REPORT.md").exists()
