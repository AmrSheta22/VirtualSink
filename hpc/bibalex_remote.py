"""Read or administer Bibalex through the existing local SSH key."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/hpc-client"))
import paramiko


def connect():
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "server_access/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("login02.c2.hpc.bibalex.org", username="alex116u1",
                   key_filename=str(ROOT / "server_access/private_key"),
                   look_for_keys=False, allow_agent=False, timeout=30)
    return client


def execute(client, command):
    _, out, _ = client.exec_command(command)
    out.channel.set_combine_stderr(True)
    for line in out:
        print(line, end="", flush=True)
    return out.channel.recv_exit_status()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Provide one remote shell command")
    sys.stdout.reconfigure(encoding="utf-8")
    client = connect()
    try:
        raise SystemExit(execute(client, sys.argv[1]))
    finally:
        client.close()
