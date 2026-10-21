import json

import pytest

from virtualsink.uptraining.memory_warmup import COMMON, matrix
from virtualsink.uptraining.recall_experiment import atomic_json
from virtualsink.uptraining.summarize_memory_warmup import summarize


def populate(root):
    atomic_json(root / "plan.json", {"common": COMMON, "matrix": matrix()})
    for arm in matrix():
        run = root / "runs" / f"{arm['name']}-seed{arm['seed']}"
        run.mkdir(parents=True)
        atomic_json(run / "complete.json", {"status": "completed", "arm": arm, "steps": 1300})
        atomic_json(run / "transfer.json", {"arm": arm, "base_unchanged": arm["warmup"] != "full_normal",
                    "calibration": {"actual_max_mass": 1e-4},
                    "after_probe": {"max_joint_logit_drift_from_original_base": .00001}})
        for step in COMMON["snapshots"]:
            policies = {}
            for name in ("joint", "writes_disabled", "reads_disabled", "zero_memory_values", "forget_zero",
                         "aux_joint", "aux_writes_disabled", "aux_forget_zero"):
                correct = 512 if name == "aux_joint" else 64
                policies[name] = {"examples": 512, "correct": correct, "accuracy": correct / 512,
                                  "ce": 1., "query_memory_mass": 1e-5}
            record = {"arm": arm, "step": step, "input_tokens": step * 32 * 26,
                      "supervised_labels": step * 32, "results": {str(d): policies for d in (24, 48, 96)}}
            atomic_json(run / f"step-{step:04d}.json", record)


def test_auxiliary_success_is_not_reported_as_normal_attention_success(tmp_path):
    populate(tmp_path)
    assert summarize(tmp_path) == {"status": "complete", "runs": 12}
    data = json.loads((tmp_path / "statistics.json").read_text())
    result = data["arms"]["aux_then_normal"]["1300"]["24"]
    assert result["aux_joint"]["mean_accuracy"] == 1.
    assert result["joint"]["mean_accuracy"] == .125


def test_partial_campaign_and_backbone_violation_are_rejected(tmp_path):
    atomic_json(tmp_path / "plan.json", {"common": COMMON, "matrix": matrix()})
    assert summarize(tmp_path)["status"] == "incomplete"
    assert not (tmp_path / "REPORT.md").exists()
    populate(tmp_path)
    path = tmp_path / "runs/aux_then_normal-seed1729/transfer.json"
    record = json.loads(path.read_text())
    record["base_unchanged"] = False
    atomic_json(path, record)
    with pytest.raises(ValueError, match="frozen backbone"):
        summarize(tmp_path)
