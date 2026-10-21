"""Freeze and submit targeted memory-reset inference."""
import hashlib,io,json,tarfile
from pathlib import Path
from bibalex_remote import connect,execute

ROOT=Path(__file__).resolve().parents[1]
NAME='qwen05-article25-reset-20261017-r1';LOCAL=ROOT/'artifacts'/NAME
REMOTE='/cluster/users/alex116u1/data/'+NAME
TRAINING='/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1'


def main():
    LOCAL.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'artifacts/qwen05-full-article-evaluation-20261017-r1/release.tar.gz') as t:
        files={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    for name in ('hpc/eval_qwen_article_reset.py','hpc/deploy_qwen_article_reset.py'):
        files[name]=(ROOT/name).read_bytes()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as t:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    plan=dict(document=25,title='Brock Lesnar',split='test',checkpoint_step=2048,
        resets=[8192,11264,11776,12288,12544],segment=256,physical_cache=128,
        intervention='Zero all recurrent cells and exposed states once at boundary; preserve physical K/V and position; writes then resume normally',
        endpoints=['same 925 predictions at positions12288..13212','full matched post-reset suffix','256-token blocks and individual token losses'],
        caveats=['One article selected after observing its gain; exploratory','Physical K/V retain earlier memory effects','Reset may disrupt state scale; it does not isolate semantic information age'],inference_only=True)
    (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    script=f'''#!/bin/bash
#SBATCH --job-name=qwen05-article-reset
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=comp002
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python -u hpc/eval_qwen_article_reset.py --root {REMOTE} --training-root {TRAINING}
'''
    (LOCAL/'eval.sh').write_bytes(script.encode())
    c=connect()
    try:
        assert execute(c,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release {REMOTE}/logs')==0
        with c.open_sftp() as s:
            for n in ('release.tar.gz','source-sha256.json','plan.json','eval.sh'):s.put(str(LOCAL/n),REMOTE+'/'+n)
            assert execute(c,f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh && ln {TRAINING}/runs/ordinary/best.pt {REMOTE}/ordinary-best.pt && bash -n {REMOTE}/eval.sh')==0
            assert execute(c,f'cd {REMOTE} && test ! -e jobs.tsv && job=$(sbatch --parsable --output=logs/eval-%j.out --error=logs/eval-%j.err eval.sh) && printf "eval\\t%s\\n" "$job" > jobs.tsv && cat jobs.tsv')==0
            s.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:c.close()


if __name__=='__main__':main()
