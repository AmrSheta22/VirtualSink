"""Collect both clusters read-only; save a local snapshot and final aggregate."""
import json
from pathlib import Path
import shlex
import statistics
import sys

from bibalex_remote import connect as connect_bibalex
from deploy_mohesr_full512 import connect, REMOTE

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/split-full512-20261009"
BIB = "/cluster/users/alex116u1/data/smollm-full512-replication-20261009"


def collect(client, root, python, gateway=False):
    script = root + "/release/hpc/report_mohesr_full512.py"
    command = " ".join(shlex.quote(x) for x in (python, script, root))
    if gateway:
        command = "ssh -o BatchMode=yes manage2 " + shlex.quote(command)
    _, out, err = client.exec_command(command)
    raw = out.read().decode()
    error = err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(error or raw)
    return json.loads(raw)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = {}
    assignment_path = LOCAL / "assignment.json"
    assignment = json.loads(assignment_path.read_text()) if assignment_path.exists() else {
        "mohesr": {"seeds": [8]}, "bibalex": {"seeds": [9, 10]}}
    for name, root, python, seeds in (
        ("mohesr", REMOTE, "/nfs/slurm/alex005/VirtualSink/venv-modern/bin/python", [8]),
        ("bibalex", BIB, "/cluster/users/alex116u1/data/phase3-k80-20260909/environment/bin/python", [9, 10]),
    ):
        seeds = assignment[name]["seeds"]
        if not seeds:
            result[name] = dict(state="stopped", checkpoint=assignment[name].get("checkpoint"), arms=[])
            continue
        client = None
        try:
            if name == "mohesr":
                client = connect("hpclogin.mohesr.gov.eg", "alex005", "private_mohesr", "known_hosts_mohesr",
                                 (ROOT / "server_access/password_mohesr.txt").read_text().strip())
            else:
                client = connect_bibalex()
            report = collect(client, root, python, gateway=name == "mohesr")
            report["arms"] = [r for r in report["arms"] if r["seed"] in seeds]
            if name == "mohesr":
                report["historical_errors"] = report["errors"].pop("pipeline-11592.err", None)
                # Preserve the resolved failure's identity without repeating its traceback.
                if report["historical_errors"]:
                    report["historical_errors"] = "11592 failed before training during the discarded microbatch-32 benchmark."
            result[name] = report
        except Exception as error:
            result[name] = {"access_error": str(error)}
        finally:
            if client:
                client.close()
    pairs = []
    seed_clusters = [(seed, cluster) for cluster in ("mohesr", "bibalex")
                     for seed in assignment[cluster]["seeds"]]
    for seed, cluster in sorted(seed_clusters):
        comparison = result.get(cluster, {}).get(f"comparison_seed{seed}")
        if comparison:
            pairs.append(dict(seed=seed, cluster=cluster, **comparison["test_comparison"],
                              disabled_intervention=comparison["disabled_intervention"]))
    if len(pairs) == 3:
        gains = [p["ce_reduction"] for p in pairs]
        aggregate = dict(seeds=pairs, mean_ce_reduction=statistics.mean(gains),
            sd_ce_reduction=statistics.stdev(gains), positive_seeds=sum(x > 0 for x in gains),
            note=("All three initialization seeds use Bibalex K80/PyTorch1.12.1 and one fixed dataset; block-bootstrap intervals are within-seed only."
                  if all(p["cluster"] == "bibalex" for p in pairs) else
                  "Across seeds, initialization is confounded with cluster/runtime; each pair is matched within one cluster. One fixed dataset; block-bootstrap intervals are within-seed only."))
        (LOCAL / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
        result["aggregate"] = aggregate
    (LOCAL / "latest-status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
