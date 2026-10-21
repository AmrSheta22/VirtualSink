"""Deploy isolated pretrained StreamingLLM screens and an initialization-only pilot."""
import hashlib,json,tarfile
from pathlib import Path
from bibalex_remote import connect,execute

REPO=Path(__file__).resolve().parents[1]
LOCAL=REPO/'artifacts/streamingllm-replication-20261013'
REMOTE='/cluster/users/alex116u1/data/streamingllm-replication-20261013'


def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for folder in ['src','hpc'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts)
    hashes={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    archive=LOCAL/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in paths:t.add(p,arcname=p.relative_to(REPO).as_posix())
    common=f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export REPLICATION_ROOT={REMOTE}
export PYTHONPATH={REMOTE}/release/src:{REMOTE}/dependencies:{REMOTE}/upstream
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
'''
    cpu='''#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --nodelist=comp023
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
'''
    gpu='''#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
'''
    prepare='#!/bin/bash\n#SBATCH --job-name=stream_rep_prepare\n#SBATCH --time=04:00:00\n'+cpu+common+'python -u hpc/prepare_streaming_replication.py\n'
    validate='#!/bin/bash\n#SBATCH --job-name=stream_rep_validate\n#SBATCH --time=02:00:00\n'+gpu+common+f'python -u hpc/run_streaming_replication.py --root {REMOTE} --validate\n'
    evaluate='#!/bin/bash\n#SBATCH --job-name=stream_rep_eval\n#SBATCH --time=48:00:00\n'+gpu+common+f'''index=${{SLURM_ARRAY_TASK_ID:?}}
labels=(pythia28 qwen15)
policies=(fifo streamingllm recompute)
label=${{labels[index/3]}}
policy=${{policies[index%3]}}
budget=400000
if [ "$policy" = recompute ]; then budget=20000; fi
python -u hpc/run_streaming_replication.py --root {REMOTE} --label "$label" --policy "$policy" --budget "$budget"
'''
    pilot='#!/bin/bash\n#SBATCH --job-name=pythia_init_pilot\n#SBATCH --time=24:00:00\n'+gpu+common+f'''arms=(standard legacy calibrated)
python -u hpc/train_pythia_init_pilot.py --root {REMOTE} --arm "${{arms[SLURM_ARRAY_TASK_ID]}}"
'''
    report='#!/bin/bash\n#SBATCH --job-name=stream_rep_report\n#SBATCH --time=00:30:00\n'+cpu+common+'python -u hpc/report_streaming_replication.py\n'
    submit=f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
for f in prepare.sh validate.sh evaluate.sh pilot.sh; do bash -n "$f"; done
prep=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$prep" > jobs.tsv
val=$(sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes --output=logs/validate-%j.out --error=logs/validate-%j.err validate.sh)
printf 'validation\\t%s\\n' "$val" >> jobs.tsv
eval=$(sbatch --parsable --array=0-5%3 --dependency="afterok:$val" --kill-on-invalid-dep=yes --output=logs/eval-%A_%a.out --error=logs/eval-%A_%a.err evaluate.sh)
printf 'evaluation-array\\t%s\\n' "$eval" >> jobs.tsv
pilot=$(sbatch --parsable --array=0-2%2 --dependency="afterok:$val" --kill-on-invalid-dep=yes --output=logs/pilot-%A_%a.out --error=logs/pilot-%A_%a.err pilot.sh)
printf 'initialization-pilot-array\\t%s\\n' "$pilot" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$eval:$pilot" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
    scripts={'prepare.sh':prepare,'validate.sh':validate,'evaluate.sh':evaluate,'pilot.sh':pilot,'report.sh':report,'submit.sh':submit}
    for name,value in scripts.items():(LOCAL/name).write_bytes(value.encode())
    c=connect()
    try:
        assert execute(c,f'set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results {REMOTE}/models')==0
        s=c.open_sftp();s.put(str(archive),REMOTE+'/release.tar.gz');s.put(str(LOCAL/'source-sha256.json'),REMOTE+'/source-sha256.json')
        assert execute(c,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp /cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh')==0
        for name in scripts:s.put(str(LOCAL/name),REMOTE+'/'+name)
        assert execute(c,f'bash {REMOTE}/submit.sh')==0
        s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
