"""Freeze and submit one isolated short recurrent-memory validation allocation."""

import argparse
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/recurrent-sink-smoke-20261013-r1"
REMOTE = "/cluster/users/alex116u1/data/recurrent-sink-smoke-20261013-r1"
SOURCE = "/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010"
RUNTIME = "/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh"


def main():
    global LOCAL, REMOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--kind", choices=("smoke", "recall"), default="smoke")
    parser.add_argument("--recall-steps", type=int, default=300)
    parser.add_argument("--recall-memory-bias", type=float, default=-2.)
    parser.add_argument("--recall-input-bias", type=float, default=-4.)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("attempt must be positive")
    LOCAL = ROOT / f"artifacts/recurrent-sink-{args.kind}-20261013-r{args.attempt}"
    REMOTE = f"/cluster/users/alex116u1/data/recurrent-sink-{args.kind}-20261013-r{args.attempt}"
    LOCAL.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "tests/test_recurrent_sink.py", ROOT / "tests/test_recurrent_training.py",
              ROOT / "hpc/smoke_recurrent_sinks.py", ROOT / "hpc/pilot_recurrent_recall.py", Path(__file__)]
    # Read each file once so hash manifest and archive contain identical bytes.
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(paths)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    archive = LOCAL / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    task = "smoke_recurrent_sinks.py" if args.kind == "smoke" else (
        f"pilot_recurrent_recall.py --steps {args.recall_steps} --memory-bias {args.recall_memory_bias} --input-bias {args.recall_input_bias}")
    script = f'''#!/bin/bash
#SBATCH --job-name=recurrent_sink_{args.kind}
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export RECURRENT_SINK_ROOT={REMOTE}
export RECURRENT_SOURCE_MANIFEST={SOURCE}/data/manifest.json
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest -q tests/test_recurrent_sink.py tests/test_recurrent_training.py
python -u hpc/{task}
'''
    (LOCAL / "smoke.sh").write_bytes(script.encode())
    plan = {"kind": "correctness-and-profile-only", "remote_root": REMOTE,
            "model": "Qwen/Qwen2.5-1.5B", "capacity": 32, "tokens_per_step": 81,
            "optimizer_steps": 2, "update_scope": "memory only", "walltime_minutes": 45,
            "isolated_source": True, "source_manifest": SOURCE + "/data/manifest.json"}
    if args.kind == "recall":
        plan.update(kind="tiny-controlled-recall", model="random one-layer NativeLM",
                    capacity=8, steps_per_arm=args.recall_steps, train_delay=24, eval_delays=[24, 48, 96],
                    initial_memory_bias=args.recall_memory_bias, initial_input_bias=args.recall_input_bias,
                    batch=32, seed=1729, update_scope="full tiny model", source_manifest=None)
    (LOCAL / "plan.json").write_text(json.dumps(plan, indent=2))
    client = connect()
    try:
        remote = shlex.quote(REMOTE)
        assert execute(client, f"set -e; test ! -e {remote}; mkdir -p {remote}/release/hpc {remote}/logs") == 0
        sftp = client.open_sftp()
        for name in ("release.tar.gz", "source-sha256.json", "plan.json", "smoke.sh"):
            sftp.put(str(LOCAL / name), REMOTE + "/" + name)
        command = f"set -e; tar -xzf {remote}/release.tar.gz -C {remote}/release; cp {shlex.quote(RUNTIME)} {remote}/release/hpc/hpc_runtime.sh; bash -n {remote}/smoke.sh"
        assert execute(client, command) == 0
        assert execute(client, f"set -e; cd {remote}; job=$(sbatch --parsable --output=logs/smoke-%j.out --error=logs/smoke-%j.err smoke.sh); printf 'smoke\\t%s\\n' \"$job\" > jobs.tsv; cat jobs.tsv") == 0
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
