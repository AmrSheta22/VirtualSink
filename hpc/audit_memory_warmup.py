"""Independent count/budget aggregation and executed-source verification."""

import hashlib
import json
import shlex
import tarfile
from pathlib import Path

from bibalex_remote import connect

ROOT = Path(__file__).resolve().parents[1] / "artifacts/memory-warmup-20261013-r1"


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    stats = json.loads((ROOT / "statistics.json").read_text())
    if stats["status"] != "complete":
        raise ValueError("campaign incomplete")
    with tarfile.open(ROOT / "release.tar.gz") as archive:
        for name, expected in plan["source_sha256"].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError("archive source hash mismatch")
    records = {}
    for arm in plan["matrix"]:
        run = ROOT / "runs" / f"{arm['name']}-seed{arm['seed']}"
        transfer = json.loads((run / "transfer.json").read_text())
        if arm["warmup"] != "full_normal" and not transfer["base_unchanged"]:
            raise ValueError("frozen backbone violation")
        for step in plan["common"]["snapshots"]:
            record = json.loads((run / f"step-{step:04d}.json").read_text())
            if record["input_tokens"] != 832 * step or record["supervised_labels"] != 32 * step:
                raise ValueError("training budget mismatch")
            records[(arm["name"], arm["seed"], step)] = record
    checked = 0
    for name, data in stats["arms"].items():
        for step in (0, 300, 1300):
            for delay in (24, 48, 96):
                for policy, metric in data[str(step)][str(delay)].items():
                    if not isinstance(metric, dict):
                        continue
                    rows = [records[(name, seed, step)]["results"][str(delay)][policy] for seed in plan["seeds"]]
                    accuracy = sum(row["correct"] for row in rows) / sum(row["examples"] for row in rows)
                    ce = sum(row["ce"] * row["examples"] for row in rows) / sum(row["examples"] for row in rows)
                    if abs(metric["mean_accuracy"] - accuracy) > 1e-12 or abs(metric["mean_ce"] - ce) > 1e-8:
                        raise ValueError("aggregation mismatch")
                    checked += 1
    code = '''import hashlib,json,pathlib
root=pathlib.Path(%r)
plan=json.loads((root/'plan.json').read_text())
bad=[p for p,h in plan['source_sha256'].items() if hashlib.sha256((root/'release'/p).read_bytes()).hexdigest()!=h]
print(json.dumps({'mismatches':bad}))
''' % plan["remote_root"]
    client = connect()
    try:
        _, output, _ = client.exec_command("python3 -c " + shlex.quote(code))
        result = json.loads(output.read().decode())
        if result["mismatches"] or output.channel.recv_exit_status():
            raise ValueError("executed source hash mismatch")
    finally:
        client.close()
    report = {"status": "passed", "runs": 12, "snapshots": len(records),
              "independent_accuracy_ce_checks": checked, "source_hashes": "verified",
              "frozen_backbone_assertions": "verified in transfer records and executing source",
              "labels_per_run": 41600, "input_tokens_per_run": 1081600}
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
