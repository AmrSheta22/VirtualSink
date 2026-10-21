"""Numerical calibration and task-budget controls for the HPC recall matrix."""

import pytest
import torch

from virtualsink.uptraining.recall_experiment import (
    COMMON, examples, initialize_mass, make_model, matrix, train_delay,
)


def test_frozen_matrix_is_factorial_and_curriculum_always_requires_eviction():
    rows = matrix()
    assert len(rows) == 24
    assert [row["index"] for row in rows] == list(range(24))
    for seed in (1729, 1730, 1731):
        subset = [row for row in rows if row["seed"] == seed]
        assert len(subset) == 8
        assert {(row["mass"], row["input_bias"]) for row in subset[:4]} == {
            (1e-4, -4.), (1e-4, -1.), (.5, -4.), (.5, -1.)}
    for row in rows:
        assert train_delay(row, 1) >= COMMON["capacity"]
        assert train_delay(row, 300) == train_delay(row, 1000) == 24
    with pytest.raises(ValueError):
        train_delay(rows[0], 0)


@pytest.mark.parametrize("target", [1e-4, .5])
def test_mass_is_measured_and_preserves_base_parameters(target):
    model = make_model(1729, -4., "cpu")
    before = {name: p.detach().clone() for name, p in model.base.named_parameters()}
    report = initialize_mass(model, 1729, target, "cpu")
    assert report["actual_max_mass"] == pytest.approx(target, rel=.01, abs=1e-6)
    assert report["conservative_calibration"]["target_mass"] == 1e-4
    for name, parameter in model.base.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    if target == 1e-4:
        assert report["max_logit_perturbation"] < .001


def test_calibration_does_not_consume_the_training_generator():
    model = make_model(1729, -1., "cpu")
    generator = torch.Generator().manual_seed(1729)
    state = generator.get_state().clone()
    initialize_mass(model, 1729, 1e-4, "cpu")
    torch.testing.assert_close(state, generator.get_state())
    tokens, targets = examples(32, 24, generator, "cpu")
    assert tokens.shape == (32, 26)
    torch.testing.assert_close(tokens[:, 0], targets)
    assert (tokens[:, 1:-1] >= 9).all() and (tokens[:, -1] == 0).all()
