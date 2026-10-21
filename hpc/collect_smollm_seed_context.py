"""Collect the six read-only prefix-deletion probes and compute paired deltas."""
import json
from pathlib import Path
from bibalex_remote import connect

OUT = Path(__file__).resolve().parents[1] / "artifacts/smollm-seed-analysis-20261016"
REMOTE = "/cluster/users/alex116u1/data/smollm-seed-analysis-20261016"


def main():
    client = connect()
    results = []
    try:
        sftp = client.open_sftp()
        for seed in (8,9,10,11,12,13):
            path = OUT / f"context-seed{seed}.json"
            try:
                sftp.get(REMOTE + f"/context-seed{seed}.json", str(path))
            except FileNotFoundError:
                continue
            value = json.loads(path.read_text())
            row = dict(seed=seed, unigram_ce=value["unigram_ce"],
                       blocks=len(value["blocks"]), tokens=value["tokens_per_condition"], arms={})
            for arm, conditions in value["arms"].items():
                row["arms"][arm] = dict(
                    normal_ce=conditions["normal"]["ce"],
                    disabled_full_ce=conditions["disabled_full"]["ce"],
                    disabled_self_only_ce=conditions["disabled_self_only"]["ce"],
                    preceding_context_removal_delta=(conditions["disabled_self_only"]["ce"]-conditions["disabled_full"]["ce"]))
            results.append(row)
    finally:
        client.close()
    (OUT / "context-analysis.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
