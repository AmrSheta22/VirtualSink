"""Freeze a synthetic retrieval evaluation; preserve all training checkpoints."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
from bibalex_remote import connect,execute

ROOT=Path(__file__).resolve().parents[1]
NAME='qwen05-long-retrieval-20261016-r1'
LOCAL=ROOT/'artifacts'/NAME
REMOTE='/cluster/users/alex116u1/data/'+NAME
TRAINING='/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1'
FOLLOWUP=False


def main():
    LOCAL.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'artifacts/qwen05-long-uptraining-20261014-r1/release.tar.gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    additions=['src/virtualsink/uptraining/recurrent_sink_optimized.py','src/virtualsink/uptraining/long_retrieval.py',
                 'hpc/eval_qwen_long_retrieval.py','hpc/deploy_qwen_long_retrieval.py','tests/test_long_retrieval.py']
    if FOLLOWUP:additions+=['hpc/prepare_qwen_retrieval_cloze.py','hpc/deploy_qwen_retrieval_cloze.py']
    for name in additions:
        files[name]=(ROOT/name).read_bytes()
    tested=json.loads((ROOT/'artifacts/qwen05-checkpoint-speed-20261016-r1/source-sha256.json').read_text())
    optimized='src/virtualsink/uptraining/recurrent_sink_optimized.py'
    assert hashlib.sha256(files[optimized]).hexdigest()==tested[optimized]
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    plan=dict(name=NAME,training_root=TRAINING,checkpoint_selection='original validation-selected best.pt, step2048, frozen before this evaluation',
        arms=['memory','memory_disabled','forget_zero','fifo'],cache_capacity=128,segment=256,batch=32,
        blocks=8,colors=8,cases_per_distance=64,filler_token_gaps=[0,64,128,256,512,1024,2048,4096,8192,16384],
        controls=['missing_answer','repeat_fact_near_query','clear_cells_at_query (memory arm only)'],
        primary_metric='balanced eight-way forced-choice accuracy; chance12.5%',
        secondary_metrics=['unrestricted next-token exact color accuracy','candidate normalized NLL','absolute answer-token NLL'],
        statistical_unit='eight key/background blocks; each block contains all eight counterfactual values; four neutral/four interfering',
        inference_only=True,source_sha256=hashes)
    (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    if FOLLOWUP:
        plan.update(query_format='cloze completion',scope='secondary format/control check declared after Q/A results; no new training or model selection',
                    parent_experiment='qwen05-long-retrieval-20261016-r1')
        (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    common=f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
'''
    prep='''#!/bin/bash
#SBATCH --job-name=qwen05-retrieval-prepare
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
'''+common+f'''python -m pytest -q tests/test_long_retrieval.py
python -u hpc/eval_qwen_long_retrieval.py --root {REMOTE} --training-root {TRAINING} --prepare
'''
    (LOCAL/'prepare.sh').write_bytes(prep.encode())
    if FOLLOWUP:
        prep=prep.replace('hpc/eval_qwen_long_retrieval.py','hpc/prepare_qwen_retrieval_cloze.py').replace(' --prepare','')
        (LOCAL/'prepare.sh').write_bytes(prep.encode())
    for arm,node in zip(plan['arms'],('comp002','comp003','comp004','comp005')):
        script=f'''#!/bin/bash
#SBATCH --job-name=qwen05-retrieval-{arm}
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist={node}
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
'''+common+f'''status 'RUNNING retrieval {arm}'
python -u hpc/eval_qwen_long_retrieval.py --root {REMOTE} --training-root {TRAINING} --arm {arm}
'''
        (LOCAL/(arm+'.sh')).write_bytes(script.encode())
    client=connect()
    try:
        if execute(client,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release/hpc {REMOTE}/logs'):
            raise RuntimeError('isolation failed')
        with client.open_sftp() as sftp:
            for name in ('release.tar.gz','source-sha256.json','plan.json','prepare.sh',*(a+'.sh' for a in plan['arms'])):
                sftp.put(str(LOCAL/name),REMOTE+'/'+name)
            if execute(client,f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh && ln {TRAINING}/runs/ordinary/best.pt {REMOTE}/ordinary-best.pt && ln {TRAINING}/runs/fifo/best.pt {REMOTE}/fifo-best.pt && bash -n {REMOTE}/prepare.sh'):
                raise RuntimeError('release setup failed')
            command=f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
prep=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$prep" > jobs.tsv
for arm in memory memory_disabled forget_zero fifo; do
job=$(sbatch --parsable --dependency=afterok:$prep --kill-on-invalid-dep=yes --output=logs/$arm-%j.out --error=logs/$arm-%j.err "$arm.sh")
printf '%s\\t%s\\n' "$arm" "$job" >> jobs.tsv
done
cat jobs.tsv
'''
            if execute(client,command):raise RuntimeError('inspect jobs.tsv before retrying')
            sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:client.close()


if __name__=='__main__':main()
