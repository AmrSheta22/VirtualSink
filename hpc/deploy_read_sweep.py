"""Freeze and submit one small inference-only read-routing diagnostic."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
NAME = "memory-read-sweep-20261013-r1"
LOCAL = ROOT / "artifacts" / NAME
REMOTE = "/cluster/users/alex116u1/data/" + NAME
RUNTIME = "/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh"


def main():
    LOCAL.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "src").rglob("*.py")) + [ROOT / "hpc/sweep_memory_reads.py", Path(__file__)]
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    with tarfile.open(LOCAL / "release.tar.gz", "w:gz") as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    script = f'''#!/bin/bash
#SBATCH --job-name=memory_read_sweep
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export READ_SWEEP_ROOT={REMOTE}
status 'RUNNING frozen read-bias sweep; inference only'
python -u hpc/sweep_memory_reads.py
'''
    (LOCAL / "sweep.sh").write_bytes(script.encode())
    client = connect()
    try:
        if execute(client, f"set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs"):
            raise RuntimeError("isolation check failed")
        sftp = client.open_sftp()
        for name in ("release.tar.gz", "source-sha256.json", "sweep.sh"):
            sftp.put(str(LOCAL / name), REMOTE + "/" + name)
        command = f'''set -e
tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release
cp {RUNTIME} {REMOTE}/release/hpc/hpc_runtime.sh
cd {REMOTE}
bash -n sweep.sh
job=$(sbatch --parsable --output=logs/sweep-%j.out --error=logs/sweep-%j.err sweep.sh)
printf 'sweep\\t%s\\n' "$job" > jobs.tsv
cat jobs.tsv
'''
        if execute(client, command):
            raise RuntimeError("submission failed; inspect jobs.tsv")
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
