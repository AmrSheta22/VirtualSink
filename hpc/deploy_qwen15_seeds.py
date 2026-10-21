"""Deploy an isolated Qwen1.5B three-pair uptraining campaign to Bibalex."""
import hashlib
import json
from pathlib import Path
import tarfile
from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/qwen15-matched-seeds-20261010"
REMOTE = "/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010"


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for folder in ["src", "tests"] for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts)
    paths += [ROOT / "hpc" / name for name in ["prepare_qwen15_seeds.py", "smoke_qwen15_seeds.py", "train_qwen15_seeds.sh"]]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    plan = dict(model="Qwen/Qwen2.5-1.5B", seeds=[1729, 1730, 1731],
                variants=["standard", "query_sink"], steps=1000, sequence_length=1024,
                window=512, microbatch=1, accumulation=8, input_tokens_per_arm=8192000,
                auxiliary="anti_collapse", gamma=.01, mode="attention", precision="fp32",
                concurrent_training_jobs=2, remote_root=REMOTE,
                scope="larger-model replication; chunk=32 and current gate-gradient-preserving native implementation; not a historical byte-for-byte replay")
    (LOCAL / "plan.json").write_text(json.dumps(plan, indent=2))
    archive = LOCAL / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    client = connect()
    try:
        assert execute(client, f"set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release {REMOTE}/logs") == 0
        sftp = client.open_sftp()
        for p in [archive, LOCAL / "plan.json", LOCAL / "source-sha256.json"]:
            sftp.put(str(p), REMOTE + "/" + p.name)
        assert execute(client, f"tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release") == 0
        runtime = '''#!/bin/bash
: "${SLURM_JOB_ID:?Slurm allocation required}"
source "$HOME/data/VirtualSink/venv/bin/activate"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
status() { echo "$(date -Is) $* job=$SLURM_JOB_ID host=$(hostname)"; }
status STARTED
status ENV_READY
(while sleep 10; do status HEARTBEAT; done) &
runtime_heartbeat=$!
trap 'code=$?; kill "$runtime_heartbeat" 2>/dev/null || true; status "FINISHED exit_code=$code"' EXIT
'''
        prepare = '''#!/bin/bash
#SBATCH --job-name=qwen15_prepare
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$QWEN_SEEDS_ROOT/release"
source hpc/hpc_runtime.sh
export PYTHONPATH="$QWEN_SEEDS_ROOT/release/src"
python -u hpc/prepare_qwen15_seeds.py
'''
        smoke = '''#!/bin/bash
#SBATCH --job-name=qwen15_smoke
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$QWEN_SEEDS_ROOT/release"
source hpc/hpc_runtime.sh
export PYTHONPATH="$QWEN_SEEDS_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python -m pytest -q tests/test_native_uptraining.py tests/test_swa_uptraining.py tests/test_uptrained_streaming.py tests/test_streaming_confirmation.py
python -u hpc/smoke_qwen15_seeds.py
'''
        submit = f'''#!/bin/bash
set -euo pipefail
export QWEN_SEEDS_ROOT={REMOTE}
cd "$QWEN_SEEDS_ROOT"
test ! -e jobs.tsv
bash -n prepare.sh smoke.sh release/hpc/train_qwen15_seeds.sh
prepare=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$prepare" > jobs.tsv
smoke=$(sbatch --parsable --dependency="afterok:$prepare" --kill-on-invalid-dep=yes --output=logs/smoke-%j.out --error=logs/smoke-%j.err smoke.sh)
printf 'smoke\\t%s\\n' "$smoke" >> jobs.tsv
train=$(sbatch --parsable --array=0-5%2 --dependency="afterok:$smoke" --kill-on-invalid-dep=yes --output=logs/train-%A_%a.out --error=logs/train-%A_%a.err release/hpc/train_qwen15_seeds.sh)
printf 'training-array\\t%s\\n' "$train" >> jobs.tsv
cat jobs.tsv
'''
        for name, content in [("release/hpc/hpc_runtime.sh", runtime), ("prepare.sh", prepare), ("smoke.sh", smoke), ("submit.sh", submit)]:
            with sftp.open(REMOTE + "/" + name, "w") as handle:
                handle.write(content)
        assert execute(client, f"bash {REMOTE}/submit.sh") == 0
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
        print("REMOTE_ROOT", REMOTE, flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
