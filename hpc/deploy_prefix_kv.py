"""Freeze and submit prefix-equivalent learned KV sinks and matched controls."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
from bibalex_remote import connect,execute

REPO=Path(__file__).resolve().parents[1]
LOCAL=REPO/'artifacts/pythia-prefix-kv-20261018'
REMOTE='/cluster/users/alex116u1/data/pythia-prefix-kv-20261018'
ASSETS='/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3'
REFERENCE='/cluster/users/alex116u1/data/pythia410-trained-streaming-20261014-r2'


def main():
    global LOCAL,REMOTE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',choices=['r2','r3','r4']);args=parser.parse_args()
    if args.attempt:
        REMOTE+='-'+args.attempt;LOCAL=REPO/'artifacts'/('pythia-prefix-kv-20261018-'+args.attempt)
    sources=['hpc/run_prefix_kv.py','tests/test_prefix_kv.py','src/virtualsink/__init__.py',
             'src/virtualsink/uptraining/__init__.py','src/virtualsink/uptraining/prefix_kv.py',
             'src/virtualsink/uptraining/neox_streaming.py',
             'src/virtualsink/attention/__init__.py','src/virtualsink/attention/query_sink.py',
             'src/virtualsink/attention/baselines.py']
    hashes={}
    for name in sources:
        dest=LOCAL/'release'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/name,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for name in sources:archive.add(LOCAL/'release'/name,arcname=name)
    common=f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src:{ASSETS}/dependencies:{ASSETS}/upstream
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
'''
    gpu='''#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp004,comp005,comp013,comp014,comp015,comp016
#SBATCH --no-requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
'''
    cpu='''#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --exclude=comp090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
'''
    call='python -u hpc/run_prefix_kv.py '
    scripts={
        'validate.sh':'#!/bin/bash\n#SBATCH --job-name=prefix_kv_gate\n#SBATCH --time=02:00:00\n'+gpu+common+'python -m pytest -q tests/test_prefix_kv.py\n'+call+f'prepare --root {REMOTE}\n',
        'train.sh':'#!/bin/bash\n#SBATCH --job-name=prefix_kv_train\n#SBATCH --time=12:00:00\n'+gpu+common+call+f'train --root {REMOTE}\n',
        'evaluate.sh':'#!/bin/bash\n#SBATCH --job-name=prefix_kv_eval\n#SBATCH --time=24:00:00\n'+gpu+common+call+f'evaluate --root {REMOTE}\n',
        'report.sh':'#!/bin/bash\n#SBATCH --job-name=prefix_kv_report\n'+cpu+common+call+f'report --root {REMOTE}\n'}
    scripts['submit.sh']=f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
for f in validate.sh train.sh evaluate.sh report.sh; do bash -n "$f"; done
gate=$(sbatch --parsable --output=logs/gate-%j.out --error=logs/gate-%j.err validate.sh)
printf 'validation\t%s\n' "$gate" > jobs.tsv
train=$(sbatch --parsable --dependency="afterok:$gate" --kill-on-invalid-dep=yes --output=logs/train-%j.out --error=logs/train-%j.err train.sh)
printf 'training\t%s\n' "$train" >> jobs.tsv
eval=$(sbatch --parsable --array=0-5%4 --dependency="afterok:$train" --kill-on-invalid-dep=yes --output=logs/eval-%A_%a.out --error=logs/eval-%A_%a.err evaluate.sh)
printf 'evaluation\t%s\n' "$eval" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$eval" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\t%s\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
    c=connect()
    try:
        assert execute(c,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results')==0
        with c.open_sftp() as s:
            for name in ['release.tar.gz','source-sha256.json']:s.put(str(LOCAL/name),REMOTE+'/'+name)
            for name,text in scripts.items():
                (LOCAL/name).write_bytes(text.encode());s.put(str(LOCAL/name),REMOTE+'/'+name)
        assert execute(c,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {REFERENCE}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/; bash {REMOTE}/submit.sh')==0
        with c.open_sftp() as s:s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
