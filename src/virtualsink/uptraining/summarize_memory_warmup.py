"""Strict completion and seed-level summaries of memory warmup transfer."""

import argparse
import json
from pathlib import Path

from .memory_warmup import ARMS, COMMON, matrix
from .recall_experiment import SEEDS, atomic_json


def summarize(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if plan["common"] != COMMON or plan["matrix"] != matrix():
        raise ValueError("frozen warmup plan mismatch")
    records, transfers, missing = {}, {}, []
    for arm in matrix():
        name = f"{arm['name']}-seed{arm['seed']}"
        run = root / "runs" / name
        paths = [run / "complete.json", run / "transfer.json", *[run / f"step-{step:04d}.json" for step in COMMON["snapshots"]]]
        missing.extend(str(path.relative_to(root)) for path in paths if not path.exists())
        if not all(path.exists() for path in paths):
            continue
        completion = json.loads(paths[0].read_text())
        if completion["status"] != "completed" or completion["arm"] != arm or completion["steps"] != COMMON["steps"]:
            raise ValueError("completion identity/budget mismatch")
        transfer = json.loads(paths[1].read_text())
        if transfer["arm"] != arm or (arm["warmup"] != "full_normal" and not transfer["base_unchanged"]):
            raise ValueError("warmup changed frozen backbone")
        if transfer["calibration"]["actual_max_mass"] > 1.01e-4:
            raise ValueError("transfer attention mass exceeds calibrated target")
        transfers[(arm["name"], arm["seed"])] = transfer
        for step, path in zip(COMMON["snapshots"], paths[2:]):
            record = json.loads(path.read_text())
            if (record["arm"] != arm or record["step"] != step or record["input_tokens"] != step * 32 * 26
                    or record["supervised_labels"] != step * 32):
                raise ValueError("snapshot identity/token/label mismatch")
            if set(record["results"]) != {"24", "48", "96"}:
                raise ValueError("missing evaluation delay")
            for policies in record["results"].values():
                if set(policies) != {"joint", "writes_disabled", "reads_disabled", "zero_memory_values", "forget_zero",
                                     "aux_joint", "aux_writes_disabled", "aux_forget_zero"}:
                    raise ValueError("missing evaluation policy")
                for metric in policies.values():
                    if metric["examples"] != 512 or not 0 <= metric["correct"] <= 512 or metric["accuracy"] != metric["correct"] / 512:
                        raise ValueError("evaluation counts mismatch")
            records[(arm["name"], arm["seed"], step)] = record
    if missing:
        atomic_json(root / "INCOMPLETE.json", {"missing": missing})
        return {"status": "incomplete", "missing_count": len(missing)}
    report = {"status": "complete", "runs": len(matrix()), "common": COMMON, "seeds": list(SEEDS), "arms": {}}
    for arm in ARMS:
        data = {}
        for step in COMMON["snapshots"]:
            delays = {}
            for delay in COMMON["evaluation_delays"]:
                rows = [records[(arm["name"], seed, step)]["results"][str(delay)] for seed in SEEDS]
                delays[str(delay)] = {policy: {"accuracy_by_seed": [row[policy]["accuracy"] for row in rows],
                                               "mean_accuracy": sum(row[policy]["correct"] for row in rows) / (512 * len(rows)),
                                               "mean_ce": sum(row[policy]["ce"] for row in rows) / len(rows)} for policy in rows[0]}
                delays[str(delay)]["query_mass_by_seed"] = [row["joint"]["query_memory_mass"] for row in rows]
            data[str(step)] = delays
        data["transfer_max_logit_drift_by_seed"] = [transfers[(arm["name"], seed)]["after_probe"]["max_joint_logit_drift_from_original_base"] for seed in SEEDS]
        report["arms"][arm["name"]] = data
    atomic_json(root / "statistics.json", report)
    lines = ["# Direct memory warmup experiment", "", "All 12 tasks completed, three seeds; random one-layer models only.", "",
             "| Arm | Auxiliary recall after warmup | Normal recall after warmup | Final normal: delay 24 | Delay 48 | Delay 96 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        data = report["arms"][arm["name"]]
        values = [data["300"]["24"]["aux_joint"]["mean_accuracy"], data["300"]["24"]["joint"]["mean_accuracy"],
                  *[data["1300"][str(delay)]["joint"]["mean_accuracy"] for delay in COMMON["evaluation_delays"]]]
        lines.append("| " + arm["name"] + " | " + " | ".join(f"{v:.1%}" for v in values) + " |")
    lines += ["", "Initial/transfer attention mass is recalibrated to 0.0001. Warmup auxiliary decoding removes joint-mass scaling and has no extra decoder parameters.",
              "Only warmup freezes the backbone/read keys/scores; normal adaptation can change them. Continued auxiliary gradients may update the backbone in adaptation.",
              "Auxiliary recall and ordinary attention recall are different outcomes; only the latter establishes successful transfer.",
              "All arms have 1,300 updates and the same ordered examples. Objectives and trainable scopes differ during warmup as specified in the frozen plan."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "complete", "runs": len(matrix())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root)))


if __name__ == "__main__":
    main()
