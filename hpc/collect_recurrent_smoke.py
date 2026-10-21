"""Read-only collection of small recurrent-memory gate logs and results."""

import argparse
import json
import shlex
from pathlib import Path

from bibalex_remote import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--kind", choices=("smoke", "recall"), default="smoke")
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("attempt must be positive")
    name = f"recurrent-sink-{args.kind}-20261013-r{args.attempt}"
    local = Path(__file__).resolve().parents[1] / "artifacts" / name
    remote = "/cluster/users/alex116u1/data/" + name
    job = (local / "jobs.tsv").read_text().strip().split("\t")[1]
    if not job.isdigit():
        raise ValueError("invalid job identity")
    client = connect()
    try:
        _, out, _ = client.exec_command(f"sacct -j {shlex.quote(job)} -P -n -o JobID,State,Elapsed,ExitCode")
        accounting = out.read().decode()
        sftp = client.open_sftp()
        for source, target in ((f"logs/smoke-{job}.out", "stdout.log"),
                               (f"logs/smoke-{job}.err", "stderr.log"), ("smoke.json", "smoke.json"),
                               ("recall.json", "recall.json")):
            try:
                sftp.stat(remote + "/" + source)
                temporary = local / (target + ".download.tmp")
                sftp.get(remote + "/" + source, str(temporary))
                temporary.replace(local / target)
            except FileNotFoundError:
                pass
        observation = {"job": job, "accounting": accounting.strip().splitlines()}
        if (local / "stdout.log").exists():
            lines = (local / "stdout.log").read_text().splitlines()
            observation["test_results"] = [line for line in lines if " passed" in line or "failed" in line]
            observation["gate_passed_marker"] = "RECURRENT_MEMORY_GPU_GATE_PASSED" in lines
            events = []
            for line in lines:
                if not line.startswith("{"):
                    continue
                try:
                    # Runtime heartbeats may share a stdout line with a JSON
                    # event. Canonical result files remain authoritative.
                    event = json.JSONDecoder().raw_decode(line)[0]
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "calibration":
                    event = {"event": "calibration", "target_mass": event["target_mass"],
                             "max_observed_mass": max(max(row) for row in event["max_mass"])}
                if event.get("event") == "recall_results":
                    event = {"event": "recall_results", "status": event["status"],
                             "recall_gate_supported": event["recall_gate_supported"]}
                events.append(event)
            observation["events"] = events
            if args.kind == "recall":
                observation["events"] = events[-2:]
        if (local / "stderr.log").exists():
            observation["stderr_tail"] = (local / "stderr.log").read_text().splitlines()[-8:]
        if (local / "smoke.json").exists() and (local / "smoke.json").stat().st_size:
            report = json.loads((local / "smoke.json").read_text())
            observation["result"] = {key: report[key] for key in (
                "status", "new_parameters", "max_logit_error", "elapsed_seconds",
                "peak_allocated_bytes", "peak_reserved_bytes")}
        if (local / "recall.json").exists():
            observation["recall"] = json.loads((local / "recall.json").read_text())
        (local / "observation.json").write_text(json.dumps(observation, indent=2))
        print(json.dumps(observation, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
