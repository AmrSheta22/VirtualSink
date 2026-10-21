"""Isolated full-article evaluation of fixed completed checkpoints."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
from bibalex_remote import connect,execute

ROOT=Path(__file__).resolve().parents[1]
NAME='qwen05-full-article-evaluation-20261017-r1'
LOCAL=ROOT/'artifacts'/NAME
REMOTE='/cluster/users/alex116u1/data/'+NAME
TRAINING='/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1'
ARMS=('memory','memory_disabled','fifo')


def main():
    LOCAL.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'artifacts/qwen05-long-retrieval-20261016-r1/release.tar.gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    for name in ('hpc/eval_qwen_full_articles.py','hpc/deploy_qwen_full_articles.py'):
        files[name]=(ROOT/name).read_bytes()
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in files.items()}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for name,content in files.items():
            info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    plan=dict(name=NAME,training_root=TRAINING,arms=ARMS,checkpoint_step=2048,
        checkpoint_selection='previously selected best.pt; no further selection',
        splits=['test','validation'],documents='all 57 articles per split, full length, longest first',
        bands=[0,4096,8192,12288,16384],physical_cache=128,segment=256,
        memory_reset='only at article boundaries',inference_only=True,
        caveats=['Post-hoc extended evaluation after earlier 4K test; test is no longer untouched',
                 'Different articles contribute to different bands; report paired per-article deltas and counts',
                 'Deep article position does not by itself identify the age of useful information'])
    (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    for arm,node in zip(ARMS,('comp002','comp003','comp004')):
        script=f'''#!/bin/bash
#SBATCH --job-name=qwen05-full-{arm}
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist={node}
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
status 'RUNNING full article evaluation {arm}'
python -u hpc/eval_qwen_full_articles.py --root {REMOTE} --training-root {TRAINING} --arm {arm}
'''
        (LOCAL/(arm+'.sh')).write_bytes(script.encode())
    c=connect()
    try:
        assert execute(c,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release {REMOTE}/logs')==0
        with c.open_sftp() as s:
            for name in ('release.tar.gz','source-sha256.json','plan.json',*(a+'.sh' for a in ARMS)):
                s.put(str(LOCAL/name),REMOTE+'/'+name)
            assert execute(c,f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh && ln {TRAINING}/runs/ordinary/best.pt {REMOTE}/ordinary-best.pt && ln {TRAINING}/runs/fifo/best.pt {REMOTE}/fifo-best.pt')==0
            command=f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
for arm in memory memory_disabled fifo; do
bash -n "$arm.sh"
job=$(sbatch --parsable --output=logs/$arm-%j.out --error=logs/$arm-%j.err "$arm.sh")
printf '%s\\t%s\\n' "$arm" "$job" >> jobs.tsv
done
cat jobs.tsv
'''
            assert execute(c,command)==0,'Inspect remote jobs.tsv before retrying'
            s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
