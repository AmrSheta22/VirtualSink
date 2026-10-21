"""Audit frozen archive, executed source hashes and collected numerical records."""
import hashlib
import json
import math
import shlex
import tarfile
from pathlib import Path
from bibalex_remote import connect

ROOT = Path(__file__).resolve().parents[1]
NAME = "memory-block-gate-20261014-r1"
LOCAL = ROOT / "artifacts" / NAME
REMOTE = "/cluster/users/alex116u1/data/" + NAME


def main():
    manifest = json.loads((LOCAL / "source-sha256.json").read_text())
    with tarfile.open(LOCAL / "release.tar.gz") as archive:
        actual = {p.name: hashlib.sha256(archive.extractfile(p).read()).hexdigest() for p in archive.getmembers()}
    if actual != manifest:
        raise ValueError("archive hash mismatch")
    client = connect()
    try:
        code = "import hashlib,json,pathlib; r=pathlib.Path(" + repr(REMOTE) + "); m=json.loads((r/'source-sha256.json').read_text()); print(json.dumps({n:hashlib.sha256((r/'release'/n).read_bytes()).hexdigest() for n in m}))"
        _, out, err = client.exec_command("python3 -c " + shlex.quote(code))
        executed = json.loads(out.read().decode())
        if out.channel.recv_exit_status() or executed != manifest:
            raise ValueError("executed source mismatch")
    finally:
        client.close()
    plan = json.loads((LOCAL / "plan.json").read_text())
    stats_path = LOCAL / "statistics.json"
    statistics = json.loads(stats_path.read_text()) if stats_path.exists() else None
    checked = 0
    for arm in plan["matrix"]:
        run = LOCAL / "runs" / f"{arm['name']}-seed{arm['seed']}"
        for step in plan["common"]["snapshots"]:
            path = run / f"step-{step:04d}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["arm"] != arm or record["step"] != step or record["input_tokens"] != step * 32 * 26 or record["supervised_labels"] != step * 32:
                raise ValueError("budget mismatch")
            for delay, policies in record["results"].items():
                if set(policies) != {"joint", "writes_disabled", "reads_disabled", "zero_memory_values", "forget_zero", "aux_joint", "aux_writes_disabled", "aux_forget_zero", "block_teacher"}:
                    raise ValueError("missing intervention")
                for policy, metric in policies.items():
                    if metric["examples"] != 512 or not 0 <= metric["correct"] <= 512 or metric["accuracy"] != metric["correct"] / 512:
                        raise ValueError("numerical count mismatch")
                    if not all(math.isfinite(value) for value in metric.values()):
                        raise ValueError("nonfinite evaluation metric")
                    if statistics is not None:
                        peers = [a for a in plan["matrix"] if a["name"] == arm["name"]]
                        seed_index = [a["seed"] for a in peers].index(arm["seed"])
                        reported = statistics["arms"][arm["name"]][str(step)][delay][policy]
                        if any(reported[key][seed_index] != value for key, value in metric.items()):
                            raise ValueError("report differs from raw snapshots")
            checked += 1
        path = run / "transfer.json"
        if path.exists():
            transfer = json.loads(path.read_text())
            if not transfer["base_unchanged"] or transfer["calibration"]["performed"]:
                raise ValueError("warmup or transfer policy violation")
    report = {"source_files_verified": len(manifest), "archive_verified": True,
              "executed_source_verified": True, "snapshots_verified": checked,
              "statistics_verified": statistics is not None,
              "expected_snapshots": 36, "complete": checked == 36}
    (LOCAL / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
