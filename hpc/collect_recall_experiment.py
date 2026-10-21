"""Collect compact progress/results without touching running experiment source."""

import argparse
import json
from pathlib import Path

from bibalex_remote import connect

ROOT = Path(__file__).resolve().parents[1]
NAME = "recurrent-init-factorial-20261013-r1"
LOCAL = ROOT / "artifacts" / NAME
REMOTE = "/cluster/users/alex116u1/data/" + NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    plan = json.loads((LOCAL / "plan.json").read_text())
    jobs = dict(line.split("\t") for line in (LOCAL / "jobs.tsv").read_text().strip().splitlines())
    if any(not job.isdigit() for job in jobs.values()):
        raise ValueError("invalid frozen job IDs")
    client = connect()
    try:
        _, output, _ = client.exec_command("sacct -j " + ",".join(jobs.values()) + " -P -n -o JobID,State,Elapsed,ExitCode")
        accounting = output.read().decode().strip().splitlines()
        sftp = client.open_sftp()
        def fetch(source, target):
            try:
                sftp.stat(REMOTE + "/" + source)
            except FileNotFoundError:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download.tmp")
            sftp.get(REMOTE + "/" + source, str(temporary))
            temporary.replace(target)
            return True
        for kind in ("validation", "report"):
            label = "validate" if kind == "validation" else "report"
            for extension in ("out", "err"):
                fetch(f"logs/{label}-{jobs[kind]}.{extension}", LOCAL / f"{label}.{extension}")
        progress = []
        for arm in plan["matrix"]:
            name = f"{arm['name']}-seed{arm['seed']}"
            directory = LOCAL / "runs" / name
            row = {"index": arm["index"], "arm": arm["name"], "seed": arm["seed"], "step": None, "complete": False}
            if fetch(f"runs/{name}/metrics.jsonl", directory / "metrics.jsonl"):
                events = []
                for line in (directory / "metrics.jsonl").read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # The last record may still be being appended.
                training = [event for event in events if event.get("event") == "train"]
                row["step"] = training[-1]["step"] if training else 0
                row["elapsed_seconds"] = training[-1]["elapsed_seconds"] if training else None
                for event in events:
                    if event.get("event") == "startup":
                        row["initial_mass"] = event["calibration"]["actual_max_mass"]
                fetch(f"runs/{name}/config.json", directory / "config.json")
            if fetch(f"runs/{name}/complete.json", directory / "complete.json"):
                row["complete"] = True
                row["elapsed_seconds"] = json.loads((directory / "complete.json").read_text())["elapsed_seconds"]
                for step in plan["common"]["snapshots"]:
                    fetch(f"runs/{name}/step-{step:04d}.json", directory / f"step-{step:04d}.json")
            # Copy tiny stderr files if available so failures are never hidden.
            path = f"logs/train-{jobs['training-array']}_{arm['index']}.err"
            if fetch(path, directory / "stderr.log"):
                errors = (directory / "stderr.log").read_text()
                if errors.strip():
                    row["stderr_tail"] = errors.splitlines()[-5:]
            if not row["complete"]:
                path = f"logs/train-{jobs['training-array']}_{arm['index']}.out"
                if fetch(path, directory / "stdout.log"):
                    row["stdout_tail"] = (directory / "stdout.log").read_text().splitlines()[-3:]
            progress.append(row)
        reports = []
        for name in ("statistics.json", "REPORT.md", "INCOMPLETE.json"):
            if fetch(name, LOCAL / name):
                reports.append(name)
        observation = {"jobs": jobs, "accounting": accounting,
                       "completed_runs": sum(row["complete"] for row in progress),
                       "expected_runs": len(progress), "progress": progress, "reports": reports}
        if (LOCAL / "validate.out").exists():
            observation["validation_tail"] = (LOCAL / "validate.out").read_text().splitlines()[-5:]
        if (LOCAL / "validate.err").exists():
            observation["validation_stderr"] = (LOCAL / "validate.err").read_text().splitlines()[-5:]
        (LOCAL / "observation.json").write_text(json.dumps(observation, indent=2))
        if args.compact:
            compact = {key: observation[key] for key in ("jobs", "completed_runs", "expected_runs", "reports")}
            compact["active_progress"] = [{key: value for key, value in row.items() if key != "stdout_tail"}
                                          for row in progress if not row["complete"] and
                                          (row["step"] is not None or row.get("stdout_tail") or row.get("stderr_tail"))]
            compact["validation_tail"] = observation.get("validation_tail")
            compact["failures"] = [line for line in accounting if any(state in line for state in
                                   ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"))]
            print(json.dumps(compact, indent=2))
        else:
            print(json.dumps(observation, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
