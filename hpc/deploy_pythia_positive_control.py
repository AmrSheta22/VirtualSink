"""Submit fixed-checkpoint factorial controls, without training or modifying existing jobs."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from bibalex_remote import connect, execute

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO/'artifacts/pythia-positive-control-20261017'
REMOTE = '/cluster/users/alex116u1/data/pythia-positive-control-20261017'
REFERENCE = '/cluster/users/alex116u1/data/pythia410-trained-streaming-20261014-r2'
ASSETS = '/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3'


def main():
    for source in [REPO/'hpc/evaluate_pythia_positive_control.py',REPO/'tests/test_pythia_positive_control.py']:
        destination = LOCAL/'release'/source.relative_to(REPO)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    files = list((LOCAL/'release').rglob('*.py'))
    hashes = {p.relative_to(LOCAL/'release').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    protocol = dict(model='EleutherAI/pythia-410m', arms=['pretrained','standard','legacy','calibrated'],
        checkpoint='Immutable pretrained base or existing matched256-step final updates; no new training',
        predictions_per_policy=65536, policies=24, total_predictions=1572864,
        cache_saved=1024, transient_attention=1025, trim='after prediction', dtype='FP32',
        data='First65536 of original PG19 test predictions; separate-book targets, persistent cache, no cross-book target',
        factors='Prefix0/4 crossed with absolute/cache-relative RoPE; trained sinks enabled/disabled within identical checkpoint',
        positive_control='Original FIFO-absolute versus prefix4-relative CE gap >0.2 over same1024:65536 targets required for each trained arm',
        additional_control='Untouched pretrained410M, same precision and held-out token stream',
        primary='Within-checkpoint sink-enabling benefit under FIFO and reduction in prefix dependence, separately for each position policy',
        success='Need substantial prefix rescue while sink disabled, reduced rescue while enabled, and competitive enabled FIFO CE; near equality without baseline dependence is uninformative',
        provenance='Checkpoint/data/model/source SHA256 pinning; real-model official-cache parity for all24 policies over1153 tokens; existing6 trained-policy trajectories reproduced within0.003 per-token CE',
        limitations='One training seed; same already-inspected held-out stream, not independent confirmation. Prior uptraining used full512 blocks without eviction. No new-architecture or eviction-aware training claim.')
    (LOCAL/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as archive:
        for path in files: archive.add(path,arcname=path.relative_to(LOCAL/'release').as_posix())
    common = f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={ASSETS}/dependencies:{ASSETS}/upstream
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
'''
    cpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=cpu\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G\n#SBATCH --time=00:30:00\n'
    gpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --exclude=comp013,comp016\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=24G\n'
    call = 'python -u hpc/evaluate_pythia_positive_control.py'
    scripts = {
        'prepare.sh':'#!/bin/bash\n#SBATCH --job-name=pythia_pos_pin\n'+cpu+common+f'{call} prepare --root {REMOTE}\n',
        'validate.sh':'#!/bin/bash\n#SBATCH --job-name=pythia_pos_check\n#SBATCH --time=02:00:00\n'+gpu+common+f'python -m pytest -q tests/test_pythia_positive_control.py\n{call} validate --root {REMOTE}\n',
        'evaluate.sh':'#!/bin/bash\n#SBATCH --job-name=pythia_pos_eval\n#SBATCH --time=06:00:00\n'+gpu+common+f'{call} evaluate --root {REMOTE}\n',
        'report.sh':'#!/bin/bash\n#SBATCH --job-name=pythia_pos_report\n'+cpu+common+f'{call} summarize --root {REMOTE}\n'}
    scripts['submit.sh'] = f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
for script in prepare.sh validate.sh evaluate.sh report.sh; do bash -n "$script"; done
pin=$(sbatch --parsable --output=logs/pin-%j.out --error=logs/pin-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$pin" > jobs.tsv
check=$(sbatch --parsable --array=0-3%4 --dependency="afterok:$pin" --kill-on-invalid-dep=yes --output=logs/check-%A_%a.out --error=logs/check-%A_%a.err validate.sh)
printf 'validation\\t%s\\n' "$check" >> jobs.tsv
eval=$(sbatch --parsable --array=0-23%6 --dependency="afterok:$check" --kill-on-invalid-dep=yes --output=logs/eval-%A_%a.out --error=logs/eval-%A_%a.err evaluate.sh)
printf 'evaluation\\t%s\\n' "$eval" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$eval" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
    client = connect()
    try:
        assert execute(client,f'set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/results {REMOTE}/validation {REMOTE}/logs')==0
        with client.open_sftp() as sftp:
            for name in ['release.tar.gz','source-sha256.json','protocol.json']:
                sftp.put(str(LOCAL/name),REMOTE+'/'+name)
            for name,content in scripts.items():
                (LOCAL/name).write_bytes(content.encode())
                with sftp.open(REMOTE+'/'+name,'w') as handle: handle.write(content)
        assert execute(client,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {REFERENCE}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/; bash {REMOTE}/submit.sh')==0
        with client.open_sftp() as sftp: sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally: client.close()


if __name__=='__main__': main()
