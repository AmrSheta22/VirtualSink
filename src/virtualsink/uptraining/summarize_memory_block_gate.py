"""Validate completion and summarize the decoder/gate factorial."""
import argparse
import json
from pathlib import Path
from .memory_block_gate import COMMON, ARMS, matrix
from .recall_experiment import atomic_json


def summarize(root):
    plan = json.loads((root / "plan.json").read_text())
    if plan["common"] != COMMON or plan["matrix"] != matrix():
        raise ValueError("plan mismatch")
    rows, missing = [], []
    for arm in matrix():
        run = root / "runs" / f"{arm['name']}-seed{arm['seed']}"
        paths = [run / "complete.json", run / "transfer.json", *[run / f"step-{s:04d}.json" for s in COMMON["snapshots"]]]
        missing.extend(str(p.relative_to(root)) for p in paths if not p.exists())
        if not all(p.exists() for p in paths):
            continue
        complete, transfer, *snapshots = [json.loads(p.read_text()) for p in paths]
        if (complete["arm"] != arm or complete["status"] != "completed" or complete["steps"] != 1300
                or complete["input_tokens"] != 1300 * 32 * 26 or complete["supervised_labels"] != 1300 * 32
                or not transfer["base_unchanged"] or transfer["calibration"]["performed"]):
            raise ValueError("completion, budget or frozen backbone mismatch")
        for step, record in zip(COMMON["snapshots"], snapshots):
            if record["arm"] != arm or record["step"] != step or record["input_tokens"] != step * 32 * 26:
                raise ValueError("snapshot mismatch")
            for policies in record["results"].values():
                for metric in policies.values():
                    if metric["examples"] != 512 or metric["accuracy"] != metric["correct"] / 512:
                        raise ValueError("count mismatch")
        rows.append({"arm": arm, "snapshots": snapshots})
    if missing:
        atomic_json(root / "INCOMPLETE.json", {"missing": missing})
        return {"status": "incomplete", "missing": len(missing)}
    report = {"status": "complete", "runs": len(rows), "common": COMMON, "arms": {}}
    lines = ["# Complete-block teacher and read-gate experiment", "", "Random one-layer recall model, three seeds; no Qwen language-model claim.", "",
             "| Arm | Normal recall 24 | 48 | 96 | Block teacher 24 | Query memory mass | Background mass |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        selected = [r for r in rows if r["arm"]["name"] == arm["name"]]
        data = {}
        for step_index, step in enumerate(COMMON["snapshots"]):
            data[str(step)] = {}
            for delay in COMMON["evaluation_delays"]:
                results = [r["snapshots"][step_index]["results"][str(delay)] for r in selected]
                data[str(step)][str(delay)] = {p: {key: [r[p][key] for r in results] for key in results[0][p]} for p in results[0]}
        report["arms"][arm["name"]] = data
        final = data["1300"]
        mean = lambda values: sum(values) / len(values)
        vals = [mean(final[str(d)]["joint"]["accuracy"]) for d in (24, 48, 96)]
        vals += [mean(final["24"]["block_teacher"]["accuracy"]), mean(final["24"]["joint"]["query_memory_mass"]), mean(final["24"]["joint"]["background_memory_mass"])]
        lines.append("| " + arm["name"] + " | " + " | ".join(f"{v:.3%}" for v in vals) + " |")
    lines += ["", "Initial mass 0.0001; no recalibration at transfer. All arms freeze the backbone for 300 steps, then train all active parameters for 1000 steps with retained teacher loss. Gate arms add balanced BCE targeting 0.9 on retrieval queries and 0.0001 on background positions. Teacher forcing is excluded from ordinary evaluation. See statistics.json for seed-level results and memory interventions."]
    atomic_json(root / "statistics.json", report)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "complete", "runs": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().root)))
