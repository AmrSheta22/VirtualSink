"""Strict completeness and descriptive paired summaries for recall factorial."""

import argparse
import json
from pathlib import Path

from .recall_experiment import ARMS, COMMON, SEEDS, atomic_json, matrix


def summarize(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if plan["matrix"] != matrix() or plan["common"] != COMMON:
        raise ValueError("summary source and frozen plan mismatch")
    snapshots = {}
    missing = []
    timings = {}
    for row in matrix():
        name = f"{row['name']}-seed{row['seed']}"
        run = root / "runs" / name
        complete_path = run / "complete.json"
        if not complete_path.exists():
            missing.append(name + "/complete.json")
            continue
        complete = json.loads(complete_path.read_text())
        if complete["status"] != "completed" or complete["arm"] != row or complete["steps"] != COMMON["steps"]:
            raise ValueError("run identity/budget mismatch: " + name)
        timings[name] = complete["elapsed_seconds"]
        for step in COMMON["snapshots"]:
            path = run / f"step-{step:04d}.json"
            if not path.exists():
                missing.append(name + "/" + path.name)
                continue
            snapshot = json.loads(path.read_text())
            if snapshot["arm"] != row or snapshot["step"] != step or snapshot["supervised_labels"] != step * COMMON["batch"]:
                raise ValueError("snapshot identity/count mismatch: " + name)
            policies = {"trained_standard"} if row["mass"] == 0 else {
                "memory", "zero_memory_values", "writes_disabled", "reads_disabled", "forget_zero"}
            if set(snapshot["results"]) != {str(delay) for delay in COMMON["evaluation_delays"]}:
                raise ValueError("evaluation delays mismatch")
            for results in snapshot["results"].values():
                if set(results) != policies:
                    raise ValueError("missing evaluation policy")
                for result in results.values():
                    n, correct = result["examples"], result["correct"]
                    if n != COMMON["evaluation_examples"] or not 0 <= correct <= n or result["accuracy"] != correct / n:
                        raise ValueError("evaluation count/accuracy mismatch")
            snapshots[(row["name"], row["seed"], step)] = snapshot
    if missing:
        atomic_json(root / "INCOMPLETE.json", {"status": "incomplete", "missing": missing})
        return {"status": "incomplete", "missing_count": len(missing)}
    summary = {"status": "complete", "runs": len(matrix()), "seeds": list(SEEDS),
               "common": COMMON, "run_seconds": timings, "arms": {}, "contrasts": {}}
    for arm in ARMS:
        records = {}
        for step in COMMON["snapshots"]:
            delays = {}
            for delay in COMMON["evaluation_delays"]:
                policy = "trained_standard" if arm["mass"] == 0 else "memory"
                rows = [snapshots[(arm["name"], seed, step)]["results"][str(delay)] for seed in SEEDS]
                accuracy = [row[policy]["accuracy"] for row in rows]
                ce = [row[policy]["ce"] for row in rows]
                entry = {"accuracy_by_seed": accuracy, "mean_accuracy": sum(accuracy) / len(accuracy),
                         "accuracy_range": [min(accuracy), max(accuracy)],
                         "ce_by_seed": ce, "mean_ce": sum(ce) / len(ce)}
                if arm["mass"]:
                    entry["ablation_mean_accuracy"] = {p: sum(row[p]["accuracy"] for row in rows) / len(rows)
                                                       for p in rows[0] if p != "memory"}
                delays[str(delay)] = entry
            records[str(step)] = delays
        summary["arms"][arm["name"]] = records
    # Paired differences use the same seed/examples and change a single factor.
    pairs = [("write_low_fixed", "low_strong_fixed", "low_weak_fixed"),
             ("write_high_fixed", "high_strong_fixed", "high_weak_fixed"),
             ("read_weak_fixed", "high_weak_fixed", "low_weak_fixed"),
             ("read_strong_fixed", "high_strong_fixed", "low_strong_fixed"),
             ("curriculum_low_weak", "low_weak_curriculum", "low_weak_fixed"),
             ("curriculum_low_strong", "low_strong_curriculum", "low_strong_fixed")]
    for name, left, right in pairs:
        contrasts = {}
        for delay in COMMON["evaluation_delays"]:
            differences = [snapshots[(left, seed, 1000)]["results"][str(delay)]["memory"]["accuracy"] -
                           snapshots[(right, seed, 1000)]["results"][str(delay)]["memory"]["accuracy"] for seed in SEEDS]
            contrasts[str(delay)] = {"difference_by_seed": differences, "mean_difference": sum(differences) / len(differences)}
        summary["contrasts"][name] = {"left": left, "right": right, "delays": contrasts}
    atomic_json(root / "statistics.json", summary)
    lines = ["# Recurrent-memory initialization experiment", "", "All 24 runs completed; three training seeds.", "",
             "Accuracy below is the mean across seeds; chance is 12.5%. These are random one-layer models.", "",
             "| Arm | Step 300: delay 24 | Step 1000: delay 24 | Delay 48 | Delay 96 |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        result = summary["arms"][arm["name"]]
        values = [result["300"]["24"]["mean_accuracy"],
                  *[result["1000"][str(delay)]["mean_accuracy"] for delay in COMMON["evaluation_delays"]]]
        lines.append("| " + arm["name"] + " | " + " | ".join(f"{value:.1%}" for value in values) + " |")
    lines += ["", "Fixed-delay arms isolate initial read mass and input-gate bias. Step 300/1000 snapshots isolate budget within the same trajectory.",
              "Curriculum comparisons change delay distribution and input-token counts, while preserving the supervised-label budget.",
              "Seed ranges, CE, probability/mass diagnostics and ablations are in the JSON files. Three seeds do not justify tight confidence claims.",
              "No pretrained-Qwen uptraining or corpus-streaming benefit is established by this experiment."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "complete", "runs": len(matrix())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root)))


if __name__ == "__main__":
    main()
