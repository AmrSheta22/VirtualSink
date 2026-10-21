"""Freeze and submit pinned pretrained Qwen0.5B full-block-teacher pilot."""
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path
from bibalex_remote import connect, execute
from virtualsink.uptraining.qwen05_teacher import COMMON

ROOT = Path(__file__).resolve().parents[1]
NAME = 'qwen05-full-block-teacher-20261014-r1'
LOCAL = ROOT/'artifacts'/NAME
REMOTE = '/cluster/users/alex116u1/data/'+NAME
RUNTIME = '/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh'


def main():
    LOCAL.mkdir(exist_ok=False)
    paths = list((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'tests'/n for n in ('test_recurrent_sink.py','test_multilayer_memory_teacher.py','test_recurrent_training.py')]
    paths += [Path(__file__), ROOT/'hpc/prepare_qwen05_teacher.py', ROOT/'hpc/report_qwen05_teacher.py']
    files = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(paths)}
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for n,b in files.items():
            entry = tarfile.TarInfo(n); entry.size=len(b)
            archive.addfile(entry,io.BytesIO(b))
    plan = {'name':NAME,'common':COMMON,'arms':['ordinary','full_block_teacher'],
        'model':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
        'remote_root':REMOTE,'source_sha256':hashes,
        'scope':'single-seed 8192-input-token-per-arm language-model feasibility screen',
        'comparison':'identical ordinary CE plus matched post-eviction targets in second forward; second forward forces memory only for teacher arm',
        'teacher':'separate causal multilayer trajectory; memory-only reads after eviction; real residual and MLP at every layer',
        'limitations':['short context128/cache32','single seed','small validation/test','no trained memory-free control','no from-scratch run']}
    (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    common = f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export QWEN_TEACHER_ROOT={REMOTE}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
'''
    cpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=cpu\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G\n#SBATCH --time=00:20:00\n'
    gpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --exclude=comp013,comp016\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=24G\n'
    scripts = {
        'prepare.sh':'#!/bin/bash\n#SBATCH --job-name=qwen05_teacher_prepare\n'+cpu+common+'python -u hpc/prepare_qwen05_teacher.py\n',
        'validate.sh':'#!/bin/bash\n#SBATCH --job-name=qwen05_teacher_gate\n#SBATCH --time=00:45:00\n'+gpu+common+f'''python -m pytest -q tests/test_recurrent_sink.py tests/test_recurrent_training.py tests/test_multilayer_memory_teacher.py
python -u -m virtualsink.uptraining.qwen05_teacher --root {REMOTE} --arm full_block_teacher --smoke
''',
        'train.sh':'#!/bin/bash\n#SBATCH --job-name=qwen05_teacher_pilot\n#SBATCH --time=04:00:00\n'+gpu+common+f'''arms=(ordinary full_block_teacher)
python -u -m virtualsink.uptraining.qwen05_teacher --root {REMOTE} --arm "${{arms[$SLURM_ARRAY_TASK_ID]}}"
''',
        'report.sh':'#!/bin/bash\n#SBATCH --job-name=qwen05_teacher_report\n'+cpu+common+'python -u hpc/report_qwen05_teacher.py\n'}
    for name,s in scripts.items():
        (LOCAL/name).write_bytes(s.encode())
    client=connect()
    try:
        q=shlex.quote(REMOTE)
        if execute(client,f'set -e; test ! -e {q}; mkdir -p {q}/release/hpc {q}/logs {q}/runs'):
            raise RuntimeError('isolation failed')
        with client.open_sftp() as sftp:
            for name in ['release.tar.gz','source-sha256.json','plan.json',*scripts]:
                sftp.put(str(LOCAL/name),REMOTE+'/'+name)
            if execute(client,f'set -e; tar -xzf {q}/release.tar.gz -C {q}/release; cp {shlex.quote(RUNTIME)} {q}/release/hpc/hpc_runtime.sh; cd {q}; bash -n prepare.sh; bash -n validate.sh; bash -n train.sh; bash -n report.sh'):
                raise RuntimeError('script validation failed')
            command=f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
prep=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$prep" > jobs.tsv
gate=$(sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes --output=logs/validate-%j.out --error=logs/validate-%j.err validate.sh)
printf 'validation\\t%s\\n' "$gate" >> jobs.tsv
train=$(sbatch --parsable --array=0-1%2 --dependency="afterok:$gate" --kill-on-invalid-dep=yes --output=logs/train-%A_%a.out --error=logs/train-%A_%a.err train.sh)
printf 'training-array\\t%s\\n' "$train" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$train" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
            if execute(client,command):
                raise RuntimeError('submission failed; inspect remote jobs.tsv before retrying')
            sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:
        client.close()


if __name__=='__main__':
    main()
