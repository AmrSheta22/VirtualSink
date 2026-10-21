"""Schedule an atomic-checkpoint handoff without changing the frozen trainer."""
import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from bibalex_remote import connect,execute

ROOT=Path(__file__).resolve().parents[1]
NAME='qwen05-optimized-handoff-20261016-r2'
LOCAL=ROOT/'artifacts'/NAME
REMOTE='/cluster/users/alex116u1/data/'+NAME
TRAINING='/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1'
NOT_BEFORE='2026-10-16T12:33:23+03:00'


def main():
    LOCAL.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'artifacts/qwen05-long-uptraining-20261014-r1/release.tar.gz') as archive:
        files={member.name:archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    for name in ('src/virtualsink/uptraining/recurrent_sink_optimized.py','hpc/qwen05_optimized_handoff.py','hpc/deploy_qwen05_optimized_handoff.py'):
        files[name]=(ROOT/name).read_bytes()
    tested=json.loads((ROOT/'artifacts/qwen05-checkpoint-speed-20261016-r1/source-sha256.json').read_text())
    optimized='src/virtualsink/uptraining/recurrent_sink_optimized.py'
    assert hashlib.sha256(files[optimized]).hexdigest()==tested[optimized]
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    plan=dict(training_root=TRAINING,legacy_primary=217102,legacy_continuation=217103,legacy_node='comp001',
        report_job=217106,not_before=NOT_BEFORE,not_before_epoch=datetime.fromisoformat(NOT_BEFORE).timestamp(),
        minimum_checkpoint_step=1152,source_manifest=REMOTE+'/source-sha256.json',
        reason='human requested optimized replacement after at least ten minutes, fully saved checkpoint verification, then running verification',
        teacher_policy='intentionally stopped; never resume',benchmark_speedup=5.219511062934031)
    (LOCAL/'handoff-plan.json').write_text(json.dumps(plan,indent=2))
    for mode in ('primary','continuation'):
        argument='' if mode=='primary' else ' --continuation'
        script=f'''#!/bin/bash
#SBATCH --job-name=qwen05-optimized-{mode}
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=comp002
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --signal=USR1@600
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
status 'RUNNING optimized {mode}'
scontrol show job -d "$SLURM_JOB_ID" > {REMOTE}/allocation-{mode}.txt
python -u hpc/qwen05_optimized_handoff.py --work {REMOTE}{argument}
'''
        (LOCAL/(mode+'.sh')).write_bytes(script.encode())
    client=connect()
    try:
        if execute(client,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release/hpc'):
            raise RuntimeError('remote isolation failed')
        with client.open_sftp() as sftp:
            for name in ('release.tar.gz','source-sha256.json','handoff-plan.json','primary.sh','continuation.sh'):
                sftp.put(str(LOCAL/name),REMOTE+'/'+name)
            if execute(client,f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh && bash -n {REMOTE}/primary.sh && bash -n {REMOTE}/continuation.sh'):
                raise RuntimeError('release setup failed')
            command=f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
first=$(sbatch --parsable --begin=now --output={TRAINING}/logs/ordinary-optimized-0-%j.out --error={TRAINING}/logs/ordinary-optimized-0-%j.err primary.sh)
printf 'ordinary-optimized-0\\t%s\\n' "$first" > jobs.tsv
printf 'ordinary-optimized-0\\t%s\\n' "$first" >> {TRAINING}/jobs.tsv
second=$(sbatch --parsable --dependency=afterok:$first --kill-on-invalid-dep=yes --output={TRAINING}/logs/ordinary-optimized-1-%j.out --error={TRAINING}/logs/ordinary-optimized-1-%j.err continuation.sh)
printf 'ordinary-optimized-1\\t%s\\n' "$second" >> jobs.tsv
printf 'ordinary-optimized-1\\t%s\\n' "$second" >> {TRAINING}/jobs.tsv
report=$(sbatch --parsable --dependency=afterany:$second --output={TRAINING}/logs/report-optimized-%j.out --error={TRAINING}/logs/report-optimized-%j.err {TRAINING}/report.sh)
printf 'report-optimized\\t%s\\n' "$report" >> jobs.tsv
printf 'report-optimized\\t%s\\n' "$report" >> {TRAINING}/jobs.tsv
cat jobs.tsv
'''
            if execute(client,command):raise RuntimeError('inspect both jobs.tsv files before any retry')
            sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
            sftp.get(TRAINING+'/jobs.tsv',str(ROOT/'artifacts/qwen05-long-uptraining-20261014-r1/jobs.tsv'))
    finally:client.close()


if __name__=='__main__':main()
