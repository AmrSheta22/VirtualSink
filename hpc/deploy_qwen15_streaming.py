"""Freeze and submit validation, twelve article jobs, and a complete-unit report."""
import hashlib
import json
from pathlib import Path
import tarfile
from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/qwen15-streaming-20261011"
REMOTE = "/cluster/users/alex116u1/data/qwen15-streaming-20261011"
TRAINING = "/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010"


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "artifacts/qwen15-matched-seeds-20261010/source-sha256.json").read_text())
    for name in ["native.py", "qwen.py", "streaming_fifo.py", "data.py"]:
        path = "src/virtualsink/uptraining/" + name
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == frozen[path], path
    files = sorted(p for directory in ["src", "tests"] for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    archive = LOCAL / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    client = connect()
    try:
        assert execute(client, f"set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results {REMOTE}/validation") == 0
        sftp = client.open_sftp()
        sftp.put(str(archive), REMOTE + "/release.tar.gz")
        sftp.put(str(LOCAL / "source-sha256.json"), REMOTE + "/source-sha256.json")
        assert execute(client, f"tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh") == 0
        # This CPU allocation pins hashes before validation and all evaluations.
        prepare = f'''#!/bin/bash
#SBATCH --job-name=q15_stream_pin
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
python - <<'PY'
import json
from pathlib import Path
from virtualsink.uptraining.data import sha256
from virtualsink.uptraining.qwen15_streaming import configs
training=Path('{TRAINING}')
runs,config=configs(training)
identity={{'checkpoint_sha256':{{variant:sha256(path/'best.pt') for variant,path in runs.items()}},'best_step':975,'model':'Qwen/Qwen2.5-1.5B','seed':1729}}
Path('{REMOTE}/checkpoint-identity.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)
PY
'''
        common = f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
'''
        gpu = '''#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
'''
        validation = '#!/bin/bash\n#SBATCH --job-name=q15_stream_check\n#SBATCH --time=02:00:00\n' + gpu + common + f'''
python -m pytest -q tests/test_qwen15_streaming.py tests/test_uptrained_streaming.py tests/test_rebased_streaming.py tests/test_streaming_confirmation.py
python -u -m virtualsink.uptraining.qwen15_streaming --training-root {TRAINING} --output {REMOTE}/validation/unit --validate-only
'''
        articles = '#!/bin/bash\n#SBATCH --job-name=q15_stream_article\n#SBATCH --time=06:00:00\n' + gpu + common + f'''
printf -v unit 'article%02d' "${{SLURM_ARRAY_TASK_ID:?}}"
python -u -m virtualsink.uptraining.qwen15_streaming --training-root {TRAINING} --output {REMOTE}/results/$unit --article-index "$SLURM_ARRAY_TASK_ID"
'''
        report = f'''#!/bin/bash
#SBATCH --job-name=q15_stream_report
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
{common}
python -u -m virtualsink.uptraining.summarize_qwen15_streaming --root {REMOTE}
'''
        submit = f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
for script in prepare.sh validate.sh articles.sh report.sh; do bash -n "$script"; done
pin=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'pin\\t%s\\n' "$pin" > jobs.tsv
validate=$(sbatch --parsable --dependency="afterok:$pin" --kill-on-invalid-dep=yes --output=logs/validate-%j.out --error=logs/validate-%j.err validate.sh)
printf 'validation\\t%s\\n' "$validate" >> jobs.tsv
array=$(sbatch --parsable --array=0-11%4 --dependency="afterok:$validate" --kill-on-invalid-dep=yes --output=logs/article-%A_%a.out --error=logs/article-%A_%a.err articles.sh)
printf 'article-array\\t%s\\n' "$array" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$array" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
        for name, content in [("prepare.sh", prepare), ("validate.sh", validation), ("articles.sh", articles), ("report.sh", report), ("submit.sh", submit)]:
            with sftp.open(REMOTE + "/" + name, "w") as handle:
                handle.write(content)
        assert execute(client, f"bash {REMOTE}/submit.sh") == 0
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
