"""Independently reaggregate raw counts and check the executed source hashes."""

import hashlib
import json
import shlex
import tarfile
from pathlib import Path

from bibalex_remote import connect

ROOT = Path(__file__).resolve().parents[1] / "artifacts/recurrent-init-factorial-20261013-r1"


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    stats = json.loads((ROOT / "statistics.json").read_text())
    if stats["status"] != "complete":
        raise ValueError("campaign not complete")
    hashes = plan["source_sha256"]
    with tarfile.open(ROOT / "release.tar.gz") as archive:
        for name, expected in hashes.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError("frozen archive hash mismatch: " + name)
    common = plan["common"]
    snapshots = {}
    for arm in plan["matrix"]:
        name = f"{arm['name']}-seed{arm['seed']}"
        run = ROOT / "runs" / name
        complete = json.loads((run / "complete.json").read_text())
        if complete["arm"] != arm or complete["status"] != "completed":
            raise ValueError("completion identity mismatch")
        for step in common["snapshots"]:
            record = json.loads((run / f"step-{step:04d}.json").read_text())
            if arm["curriculum"]:
                expected_tokens = common["batch"] * sum(10 + 16 * min(i - 1, 299) // 299 for i in range(1, step + 1))
            else:
                expected_tokens = common["batch"] * 26 * step
            if record["input_tokens"] != expected_tokens or record["supervised_labels"] != step * common["batch"]:
                raise ValueError("input/label budget mismatch: " + name)
            snapshots[(arm["name"], arm["seed"], step)] = record
    comparisons = 0
    for name, steps in stats["arms"].items():
        policy = "trained_standard" if name.startswith("standard") else "memory"
        for step, delays in steps.items():
            for delay, aggregate in delays.items():
                rows = [snapshots[(name, seed, int(step))]["results"][delay][policy] for seed in plan["seeds"]]
                mean_accuracy = sum(row["correct"] for row in rows) / sum(row["examples"] for row in rows)
                mean_ce = sum(row["ce"] * row["examples"] for row in rows) / sum(row["examples"] for row in rows)
                if abs(mean_accuracy - aggregate["mean_accuracy"]) > 1e-12 or abs(mean_ce - aggregate["mean_ce"]) > 1e-8:
                    raise ValueError("independent aggregation mismatch")
                comparisons += 1
    code = '''import hashlib,json,pathlib
root=pathlib.Path(%r)
plan=json.loads((root/'plan.json').read_text())
bad=[name for name,expected in plan['source_sha256'].items() if hashlib.sha256((root/'release'/name).read_bytes()).hexdigest()!=expected]
print(json.dumps({'source_mismatches':bad}))
''' % plan["remote_root"]
    client = connect()
    try:
        _, output, error = client.exec_command("python3 -c " + shlex.quote(code))
        result = json.loads(output.read().decode())
        if output.channel.recv_exit_status() != 0 or result["source_mismatches"]:
            raise ValueError("executed release source verification failed")
    finally:
        client.close()
    audit = {"status": "passed", "runs": len(plan["matrix"]), "snapshots": len(snapshots),
             "independent_mean_checks": comparisons, "archive_source_hashes": "verified",
             "executed_source_hashes": "verified", "input_and_label_budgets": "verified",
             "curriculum_input_tokens": 750432, "fixed_input_tokens": 832000,
             "curriculum_limit": "delay, input token budget, and generated target order differ from fixed arms"}
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
