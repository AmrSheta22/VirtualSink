"""Reject partial campaigns and keep paired recall aggregation reproducible."""

import json

import pytest

from virtualsink.uptraining.recall_experiment import COMMON, atomic_json, matrix
from virtualsink.uptraining.summarize_recall_experiment import summarize


def populate(root):
    rows = matrix()
    atomic_json(root / "plan.json", {"common": COMMON, "matrix": rows})
    for row in rows:
        run = root / "runs" / f"{row['name']}-seed{row['seed']}"
        run.mkdir(parents=True)
        atomic_json(run / "complete.json", {"status": "completed", "arm": row, "steps": 1000, "elapsed_seconds": 100.})
        policies = ["trained_standard"] if row["mass"] == 0 else [
            "memory", "zero_memory_values", "writes_disabled", "reads_disabled", "forget_zero"]
        for step in COMMON["snapshots"]:
            correct = 64 if row["mass"] == 0 else 256 + 32 * (row["seed"] - 1729)
            results = {str(delay): {policy: {"examples": 512, "correct": correct,
                                            "accuracy": correct / 512, "ce": 1.}
                                   for policy in policies} for delay in COMMON["evaluation_delays"]}
            atomic_json(run / f"step-{step:04d}.json", {"arm": row, "step": step,
                        "supervised_labels": step * 32, "results": results})


def test_incomplete_campaign_never_creates_full_report(tmp_path):
    atomic_json(tmp_path / "plan.json", {"common": COMMON, "matrix": matrix()})
    result = summarize(tmp_path)
    assert result == {"status": "incomplete", "missing_count": 24}
    assert (tmp_path / "INCOMPLETE.json").exists()
    assert not (tmp_path / "statistics.json").exists()
    assert not (tmp_path / "REPORT.md").exists()


def test_complete_summary_keeps_seeds_and_uses_paired_differences(tmp_path):
    populate(tmp_path)
    assert summarize(tmp_path) == {"status": "complete", "runs": 24}
    report = json.loads((tmp_path / "statistics.json").read_text())
    actual = report["arms"]["low_weak_fixed"]["1000"]["24"]
    assert actual["accuracy_by_seed"] == [.5, .5625, .625]
    assert actual["mean_accuracy"] == .5625
    assert report["contrasts"]["write_low_fixed"]["delays"]["24"]["difference_by_seed"] == [0., 0., 0.]


def test_malformed_accuracy_counts_fail_aggregation(tmp_path):
    populate(tmp_path)
    path = tmp_path / "runs/low_weak_fixed-seed1729/step-1000.json"
    data = json.loads(path.read_text())
    data["results"]["24"]["memory"]["correct"] = 512
    atomic_json(path, data)
    with pytest.raises(ValueError, match="count/accuracy"):
        summarize(tmp_path)
