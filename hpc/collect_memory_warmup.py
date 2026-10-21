"""Reuse read-only array collection and additionally copy transfer diagnostics."""

import json

import collect_recall_experiment as collector
from bibalex_remote import connect


def main():
    collector.NAME = "memory-warmup-20261013-r1"
    collector.LOCAL = collector.ROOT / "artifacts" / collector.NAME
    collector.REMOTE = "/cluster/users/alex116u1/data/" + collector.NAME
    collector.main()
    plan = json.loads((collector.LOCAL / "plan.json").read_text())
    client = connect()
    try:
        sftp = client.open_sftp()
        transfers = []
        for arm in plan["matrix"]:
            name = f"{arm['name']}-seed{arm['seed']}"
            remote = collector.REMOTE + f"/runs/{name}/transfer.json"
            try:
                sftp.stat(remote)
            except FileNotFoundError:
                continue
            local = collector.LOCAL / "runs" / name / "transfer.json"
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_suffix(".download.tmp")
            sftp.get(remote, str(temporary))
            temporary.replace(local)
            data = json.loads(local.read_text())
            transfers.append({"arm": name, "base_unchanged": data["base_unchanged"],
                              "after_mass": data["calibration"]["actual_max_mass"],
                              "logit_drift": data["after_probe"]["max_joint_logit_drift_from_original_base"],
                              "aux_recall_before_transfer": data["before"]["results"]["24"]["aux_joint"]["accuracy"],
                              "joint_recall_before_transfer": data["before"]["results"]["24"]["joint"]["accuracy"]})
        (collector.LOCAL / "transfers.json").write_text(json.dumps(transfers, indent=2))
        print(json.dumps({"transfers": transfers}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
