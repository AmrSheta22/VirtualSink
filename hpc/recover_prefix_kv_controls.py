"""Recover only the two controls that never started on comp004."""
import json,shlex
from pathlib import Path
from bibalex_remote import connect,execute

ROOT='/cluster/users/alex116u1/data/pythia-prefix-kv-20261018-r4'
LOCAL=Path(__file__).resolve().parents[1]/'artifacts/pythia-prefix-kv-20261018-r4'


def main():
    c=connect()
    try:
        verify=f'''import json,hashlib
from pathlib import Path
import numpy as np
r=Path({ROOT!r})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in json.loads((r/'source-sha256.json').read_text()).items():
 assert sha(r/'release'/name)==expected,name
identity=json.loads((r/'identity.json').read_text());training=json.loads((r/'training-summary.json').read_text())
assert sha(r/'initial.pt')==identity['initial_sha256']
assert sha(r/'final.pt')==training['final_sha256']
for name in ['validation.json','trained-validation.json']:
 v=json.loads((r/name).read_text());assert v['tokens']==1153 and v['oracle_max_logit_error']==0
for policy in ['fifo','natural','trained','zero_value']:
 f=r/'results'/policy;c=json.loads((f/'complete.json').read_text());x=np.load(f/'losses.npy')
 assert x.shape==(400000,) and np.isfinite(x).all()
 assert abs(float(x.mean())-c['ce'])<1e-8 and abs(float(x[1024:].mean())-c['post_eviction_ce'])<1e-8
for policy in ['fixed','frozen']:assert not (r/'results'/policy).exists()
for task in [2,3]:
 for suffix in ['out','err']:assert (r/'logs'/f'eval-217416_{{task}}.{{suffix}}').stat().st_size==0
print('SOURCE_CHECKPOINTS_GATES_AND_COMPLETED_RESULTS_VERIFIED')
'''
        python='/cluster/users/alex116u1/data/phase3-k80-20260909/environment/bin/python'
        deps='/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3/dependencies'
        assert execute(c,'PYTHONPATH='+deps+' '+python+' -c '+shlex.quote(verify))==0
        command=f'''set -euo pipefail
cd {ROOT}
test ! -e jobs-before-control-recovery.tsv
test "$(squeue -h -j 217416_2 -o %T)" = RUNNING
test "$(squeue -h -j 217417 -o %T)" = PENDING
cp jobs.tsv jobs-before-control-recovery.tsv
scancel 217416_2 217417
printf 'validation\t217414\ntraining\t217415\noriginal-evaluation\t217416\n' > jobs.tsv
controls=$(sbatch --parsable --array=2-3%2 --no-requeue --nodelist=comp003 --exclude=comp004,comp005,comp013,comp014,comp015,comp016 --output=logs/recovery-eval-%A_%a.out --error=logs/recovery-eval-%A_%a.err evaluate.sh)
printf 'recovered-fixed-frozen-controls\t%s\n' "$controls" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$controls" --kill-on-invalid-dep=yes --output=logs/recovery-report-%j.out --error=logs/recovery-report-%j.err report.sh)
printf 'report\t%s\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
        assert execute(c,command)==0
        with c.open_sftp() as s:
            for name in ['jobs.tsv','jobs-before-control-recovery.tsv']:s.get(ROOT+'/'+name,str(LOCAL/name))
    finally:c.close()


if __name__=='__main__':main()
