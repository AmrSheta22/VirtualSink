"""Transfer a frozen release and the original token data, then submit Slurm jobs.

Credentials stay local. Existing experiments cannot be overwritten or resubmitted.
"""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/hpc-client"))
import paramiko

REMOTE = "/nfs/slurm/alex005/VirtualSink/experiments/smollm-full512-20261009"
LOCAL = ROOT / "artifacts/mohesr-full512-20261009"


def connect(host, user, key, known, password=None):
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "server_access" / known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(host, username=user, key_filename=str(ROOT / "server_access" / key),
                   password=password, look_for_keys=False, allow_agent=False,
                   timeout=30, auth_timeout=30)
    return client


def command(client, script):
    _, out, _ = client.exec_command("ssh -o BatchMode=yes manage2 " + shlex.quote(script))
    out.channel.set_combine_stderr(True)
    lines = []
    for line in out:
        print(line, end="", flush=True)
        lines.append(line)
    if out.channel.recv_exit_status():
        raise RuntimeError("Remote command failed")
    return "".join(lines)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    password = (ROOT / "server_access/password_mohesr.txt").read_text().strip()
    mohesr = connect("hpclogin.mohesr.gov.eg", "alex005", "private_mohesr", "known_hosts_mohesr", password)
    bibalex = None
    try:
        command(mohesr, f"set -eu; test ! -e {REMOTE}; mkdir -p {REMOTE}/data {REMOTE}/release {REMOTE}/logs")
        target = mohesr.open_sftp()
        target.stat(REMOTE)  # Shared NFS must be accessible from the gateway.
        bibalex = connect("login02.c2.hpc.bibalex.org", "alex116u1", "private_key", "known_hosts")
        origin = bibalex.open_sftp()
        old = "/cluster/users/alex116u1/data/smollm-full512-20260913/data"
        data = LOCAL / "data"
        data.mkdir(exist_ok=True)
        origin.get(old + "/manifest.json", str(data / "manifest.json"))
        raw = (data / "manifest.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == "0d8cdb9cb8a074864e481bdec43c5e33f04f69bb2ca01fc0f0224f4baa121cdb"
        manifest = json.loads(raw)
        files = {"manifest.json": hashlib.sha256(raw).hexdigest()}
        for spec in manifest["sources"].values():
            for split in spec["splits"].values():
                files[split["file"]] = split["sha256"]
        for split, spec in manifest["splits"].items():
            files[split + "-order.npy"] = spec["order_sha256"]
        for name, digest in files.items():
            path = data / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                origin.get(old + "/" + name, str(path))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            target.put(str(path), REMOTE + "/data/" + name)
            print("Transferred verified", name, path.stat().st_size, flush=True)
        release = LOCAL / "release.tar.gz"
        hashes = {}
        with tarfile.open(release, "w:gz") as archive:
            for folder in ("src", "tests", "hpc"):
                for path in sorted((ROOT / folder).rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".sh") and "__pycache__" not in path.parts:
                        name = path.relative_to(ROOT).as_posix()
                        archive.add(path, arcname=name)
                        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ("pyproject.toml", "README.md"):
                archive.add(ROOT / name, arcname=name)
        (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
        target.put(str(release), REMOTE + "/release.tar.gz")
        target.put(str(LOCAL / "source-sha256.json"), REMOTE + "/source-sha256.json")
        scripts = {
            "pipeline.sh": """#!/bin/bash
#SBATCH --job-name=smol512_3seeds
#SBATCH --partition=gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
source /nfs/slurm/alex005/VirtualSink/venv-modern/bin/activate
export PYTHONPATH="$MOHESR_EXPERIMENT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$MOHESR_EXPERIMENT/release"
if [ ! -s "$MOHESR_EXPERIMENT/selected-microbatch.txt" ]; then
  python -m pytest -q
  python -u hpc/mohesr_full512_smoke.py
  python -u hpc/benchmark_mohesr_batches.py
fi
exec bash hpc/run_mohesr_full512.sh
""",
            "compare.sh": """#!/bin/bash
#SBATCH --job-name=smol512_compare
#SBATCH --partition=gpu5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source /nfs/slurm/alex005/VirtualSink/venv-modern/bin/activate
export PYTHONPATH="$MOHESR_EXPERIMENT/release/src"
python -m virtualsink.pretraining.compare --root "$MOHESR_EXPERIMENT/seeds/seed$1"
""",
            "aggregate.sh": """#!/bin/bash
#SBATCH --job-name=smol512_summary
#SBATCH --partition=gpu5
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
source /nfs/slurm/alex005/VirtualSink/venv-modern/bin/activate
python "$MOHESR_EXPERIMENT/release/hpc/summarize_mohesr_full512.py"
""",
        }
        for name, script in scripts.items():
            with target.open(REMOTE + "/" + name, "w") as handle:
                handle.write(script)
        submit = f"""set -euo pipefail
export MOHESR_EXPERIMENT={REMOTE}
cd {REMOTE}
tar -xzf release.tar.gz -C release
for script in pipeline.sh release/hpc/train_mohesr_full512.sh release/hpc/run_mohesr_full512.sh; do bash -n "$script"; done
test ! -e jobs.tsv
train=$(sbatch --parsable --output=logs/pipeline-%j.out --error=logs/pipeline-%j.err pipeline.sh)
printf 'pipeline\\t%s\\n' "$train" > jobs.tsv
cat jobs.tsv
"""
        # Store the exact submission protocol before launching jobs.
        with target.open(REMOTE + "/submit.sh", "w") as handle:
            handle.write(submit)
        result = command(mohesr, f"bash {REMOTE}/submit.sh")
        (LOCAL / "jobs.tsv").write_text(result)
        print("EXPERIMENT", REMOTE, flush=True)
    finally:
        if bibalex:
            bibalex.close()
        mohesr.close()


if __name__ == "__main__":
    main()
