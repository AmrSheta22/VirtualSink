"""Read-only collector for the 70-update, warmup-10 experiment."""
import json
import shlex
import statistics
from pathlib import Path
from bibalex_remote import connect

LOCAL = Path(__file__).resolve().parents[1] / "artifacts/smollm-warm70-20261014"
REMOTE = "/cluster/users/alex116u1/data/smollm-warm70-20261014"


def main():
    client = connect()
    try:
        command = " ".join(shlex.quote(x) for x in (
            "/cluster/users/alex116u1/data/phase3-k80-20260909/environment/bin/python",
            REMOTE + "/report.py", REMOTE))
        _, out, err = client.exec_command(command)
        raw = out.read().decode()
        error = err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error or raw)
        report = json.loads(raw)
    finally:
        client.close()
    pairs = []
    for seed in (11, 12, 13):
        value = report.get(f"comparison_seed{seed}")
        if value:
            pairs.append(dict(seed=seed, **value["test_comparison"],
                              disabled_intervention=value["disabled_intervention"]))
    if len(pairs) == 3:
        gains = [r["ce_reduction"] for r in pairs]
        aggregate = dict(seeds=pairs, mean_ce_reduction=statistics.mean(gains),
                         sd_ce_reduction=statistics.stdev(gains), positive_seeds=sum(g > 0 for g in gains),
                         steps=70, warmup_steps=10,
                         note="Three initialization seeds, one fixed dataset and runtime. Both warmup and training budget changed from the previous experiment.")
        (LOCAL / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
        report["aggregate"] = aggregate
    (LOCAL / "latest-status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
