"""Run commands through MOHESR's manage2 gateway without printing credentials.

Usage: python hpc/mohesr_remote.py 'squeue -u alex005'
Requires paramiko (or the local artifacts/hpc-client installation).
"""
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/hpc-client"))
import paramiko


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Provide one remote shell command")
    credentials = ROOT / "server_access"
    known_hosts = credentials / "known_hosts_mohesr"
    client = paramiko.SSHClient()
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            "hpclogin.mohesr.gov.eg", username="alex005",
            key_filename=str(credentials / "private_mohesr"),
            password=(credentials / "password_mohesr.txt").read_text().strip(),
            look_for_keys=False, allow_agent=False, timeout=25, auth_timeout=30,
        )
        client.save_host_keys(str(known_hosts))
        command = "ssh -o BatchMode=yes manage2 " + shlex.quote(sys.argv[1])
        _, stdout, stderr = client.exec_command(command)
        # Combine streams to prevent blocking on a full stderr window.
        stdout.channel.set_combine_stderr(True)
        for line in stdout:
            print(line, end="", flush=True)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
