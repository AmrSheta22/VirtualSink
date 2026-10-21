"""Freeze replay-plus-streaming jobs without modifying existing releases."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from bibalex_remote import connect,execute

REPO=Path(__file__).resolve().parents[1]
OLD='/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3'
REMOTE='/cluster/users/alex116u1/data/pythia410-trained-streaming-20261014'
LOCAL=REPO/'artifacts/pythia410-trained-streaming-20261014'


def main():
    global REMOTE,LOCAL
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',choices=['r2']);args=parser.parse_args()
    if args.attempt:
        REMOTE+='-'+args.attempt;LOCAL=REPO/'artifacts'/('pythia410-trained-streaming-20261014-'+args.attempt)
    LOCAL.mkdir(parents=True,exist_ok=True)
    files=sorted(p for folder in ['src','hpc'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts)
    (LOCAL/'source-sha256.json').write_text(json.dumps({p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    archive=LOCAL/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=p.relative_to(REPO).as_posix())
    original=(REPO/'artifacts/streamingllm-replication-20261013-r3/pilot.sh').read_text()
    job=original.replace(OLD+'/release',REMOTE+'/release').replace('--job-name=pythia_init_pilot','--job-name=pythia_trained_stream').replace('24:00:00','48:00:00')
    job=job.replace('--root '+OLD+' --arm','--root '+REMOTE+' --save-final --arm')
    job+='python -u hpc/stream_pythia_pilot.py --root '+REMOTE+' --reference-root '+OLD+' --arm "${arms[SLURM_ARRAY_TASK_ID]}"\n'
    report=f'''#!/bin/bash
#SBATCH --job-name=pythia_stream_report
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --nodelist=comp067
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={OLD}/dependencies
export PILOT_STREAM_ROOT={REMOTE} PILOT_REFERENCE_ROOT={OLD}
python -u hpc/report_pythia_pilot_streaming.py
'''
    submit=f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
replay=$(sbatch --parsable --array=0-2%3 --output=logs/pilot-stream-%A_%a.out --error=logs/pilot-stream-%A_%a.err pilot-stream.sh)
printf 'replay-and-stream-array\t%s\n' "$replay" > jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$replay:217043" --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'combined-report\t%s\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
    for name,value in [('pilot-stream.sh',job),('report.sh',report),('submit.sh',submit)]:
        (LOCAL/name).write_bytes(value.encode())
    c=connect()
    try:
        assert execute(c,f'set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results; ln -s {OLD}/data {REMOTE}/data; cp {OLD}/manifest.json {REMOTE}/manifest.json')==0
        with c.open_sftp() as s:
            s.put(str(archive),REMOTE+'/release.tar.gz');s.put(str(LOCAL/'source-sha256.json'),REMOTE+'/source-sha256.json')
            for name in ['pilot-stream.sh','report.sh','submit.sh']:s.put(str(LOCAL/name),REMOTE+'/'+name)
        assert execute(c,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {OLD}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh; bash -n {REMOTE}/pilot-stream.sh; bash -n {REMOTE}/report.sh; bash {REMOTE}/submit.sh')==0
        with c.open_sftp() as s:s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
