"""Read-only collection of the isolated block/gate factorial."""
import collect_recall_experiment as collector
from bibalex_remote import connect

collector.NAME = "memory-block-gate-20261014-r1"
collector.LOCAL = collector.ROOT / "artifacts" / collector.NAME
collector.REMOTE = "/cluster/users/alex116u1/data/" + collector.NAME

if __name__ == "__main__":
    collector.main()
    import json
    plan = json.loads((collector.LOCAL / "plan.json").read_text())
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for arm in plan["matrix"]:
                name = f"{arm['name']}-seed{arm['seed']}"
                try:
                    sftp.get(collector.REMOTE + f"/runs/{name}/transfer.json",
                             str(collector.LOCAL / "runs" / name / "transfer.json"))
                except FileNotFoundError:
                    pass
    finally:
        client.close()
