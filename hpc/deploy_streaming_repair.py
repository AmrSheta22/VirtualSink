"""Preserve failed campaign; submit a fresh release using its pinned prepared assets."""
import hashlib,json,tarfile,argparse
from pathlib import Path
from bibalex_remote import connect,execute

REPO=Path(__file__).resolve().parents[1]
OLD='/cluster/users/alex116u1/data/streamingllm-replication-20261013'
REMOTE=OLD+'-r2'
LOCAL=REPO/'artifacts/streamingllm-replication-20261013-r2'


def main():
    global REMOTE,LOCAL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',choices=['r2','r3'],default='r2')
    args=parser.parse_args()
    REMOTE=OLD+'-'+args.attempt
    LOCAL=REPO/('artifacts/streamingllm-replication-20261013-'+args.attempt)
    LOCAL.mkdir(parents=True,exist_ok=True)
    files=sorted(p for folder in ['src','hpc'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts)
    hashes={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    archive=LOCAL/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=p.relative_to(REPO).as_posix())
    scripts={}
    original=REPO/'artifacts/streamingllm-replication-20261013'
    for name in ['prepare.sh','validate.sh','evaluate.sh','pilot.sh','report.sh','submit.sh']:
        value=(original/name).read_text(encoding='utf-8').replace(OLD,REMOTE)
        if name in ['prepare.sh','report.sh']:
            value=value.replace('#SBATCH --nodelist=comp023','#SBATCH --nodelist=comp067').replace('#SBATCH --cpus-per-task=4','#SBATCH --cpus-per-task=2').replace('#SBATCH --mem=24G','#SBATCH --mem=8G')
        if name=='prepare.sh':value=value.replace('hpc/prepare_streaming_replication.py','hpc/repair_streaming_replication.py')
        scripts[name]=value
        (LOCAL/name).write_bytes(value.encode())
    c=connect()
    try:
        assert execute(c,f'set -e; test ! -e {REMOTE}; test -e {OLD}/manifest.json; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results; ln -s {OLD}/data {REMOTE}/data; ln -s {OLD}/upstream {REMOTE}/upstream; cp {OLD}/manifest.json {REMOTE}/manifest.json')==0
        s=c.open_sftp()
        s.put(str(archive),REMOTE+'/release.tar.gz');s.put(str(LOCAL/'source-sha256.json'),REMOTE+'/source-sha256.json')
        assert execute(c,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {OLD}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh')==0
        for name in scripts:s.put(str(LOCAL/name),REMOTE+'/'+name)
        assert execute(c,'bash '+REMOTE+'/submit.sh')==0
        s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
