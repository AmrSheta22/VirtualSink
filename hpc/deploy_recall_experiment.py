"""Freeze a 24-run, three-seed recall study and submit gated Slurm jobs."""

import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

from bibalex_remote import connect, execute
from virtualsink.uptraining.recall_experiment import COMMON, matrix

ROOT = Path(__file__).resolve().parents[1]
NAME = "recurrent-init-factorial-20261013-r1"
LOCAL = ROOT / "artifacts" / NAME
REMOTE = "/cluster/users/alex116u1/data/" + NAME
RUNTIME = "/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh"


def main():
    LOCAL.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "tests" / name for name in ("test_recurrent_sink.py", "test_recurrent_training.py", "test_recall_experiment.py")]
    paths += [Path(__file__)]
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(paths)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    with tarfile.open(LOCAL / "release.tar.gz", "w:gz") as archive:
        for name, data in contents.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    plan = {"name": NAME, "kind": "random-one-layer-recall-initialization-study",
            "remote_root": REMOTE, "common": COMMON, "matrix": matrix(),
            "concurrency": 6, "source_sha256": hashes, "seeds": [1729, 1730, 1731],
            "primary": "held-out delay-24 accuracy and CE at step 1000; seed-paired contrasts",
            "secondary": "step-300 learning, longer delays, memory ablations and query read mass",
            "decision": "use measured low-mass recall and intervention sensitivity before pretrained uptraining",
            "limits": ["toy architecture", "three seeds", "curriculum changes input-token budget",
                       "strong-mass arm is diagnostic, not a pretrained initialization recommendation"]}
    (LOCAL / "plan.json").write_text(json.dumps(plan, indent=2))
    shared = f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
'''
    gpu = '''#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
'''
    scripts = {
        "validate.sh": "#!/bin/bash\n#SBATCH --job-name=memory_init_validate\n#SBATCH --time=00:15:00\n" + gpu + shared + '''status 'RUNNING recurrent initialization validation'
python -m pytest -q tests/test_recurrent_sink.py tests/test_recurrent_training.py tests/test_recall_experiment.py
''',
        "train.sh": "#!/bin/bash\n#SBATCH --job-name=memory_init_factorial\n#SBATCH --time=00:30:00\n" + gpu + shared + f'''status 'RUNNING recall array index='"$SLURM_ARRAY_TASK_ID"
python -u -m virtualsink.uptraining.recall_experiment --root {REMOTE} --index "$SLURM_ARRAY_TASK_ID"
''',
        "report.sh": '''#!/bin/bash
#SBATCH --job-name=memory_init_report
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
''' + shared + f'''status 'RUNNING strict recall report'
python -u -m virtualsink.uptraining.summarize_recall_experiment --root {REMOTE}
'''}
    for name, script in scripts.items():
        (LOCAL / name).write_bytes(script.encode())
    client = connect()
    try:
        remote = shlex.quote(REMOTE)
        if execute(client, f"set -e; test ! -e {remote}; mkdir -p {remote}/release/hpc {remote}/logs {remote}/runs"):
            raise RuntimeError("remote isolation check failed")
        sftp = client.open_sftp()
        for name in ("release.tar.gz", "source-sha256.json", "plan.json", *scripts):
            sftp.put(str(LOCAL / name), REMOTE + "/" + name)
        command = f"set -e; tar -xzf {remote}/release.tar.gz -C {remote}/release; cp {shlex.quote(RUNTIME)} {remote}/release/hpc/hpc_runtime.sh; cd {remote}; bash -n validate.sh; bash -n train.sh; bash -n report.sh"
        if execute(client, command):
            raise RuntimeError("release or script validation failed")
        submit = f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
gate=$(sbatch --parsable --output=logs/validate-%j.out --error=logs/validate-%j.err validate.sh)
printf 'validation\\t%s\\n' "$gate" > jobs.tsv
train=$(sbatch --parsable --array=0-23%6 --dependency="afterok:$gate" --kill-on-invalid-dep=yes --output=logs/train-%A_%a.out --error=logs/train-%A_%a.err train.sh)
printf 'training-array\\t%s\\n' "$train" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$train" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
        if execute(client, submit):
            raise RuntimeError("submission failed; inspect remote jobs.tsv before retrying")
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
