"""Recover the node-failed validation with unchanged scientific source."""
import json
from bibalex_remote import connect, execute
from deploy_pythia_positive_control import LOCAL, REMOTE


def main():
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for arm,count in [('pretrained',4),('standard',4),('calibrated',8)]:
                with sftp.open(REMOTE+'/validation/'+arm+'.json') as handle:
                    value = json.load(handle)
                assert len(value) == count
                assert all(row['tokens']==1153 and row['maximum_absolute_logit_error']<.003 for row in value.values())
        command = f'''set -euo pipefail
cd {REMOTE}
test ! -e jobs-before-node-recovery.tsv
test ! -e validation/legacy.json
test "$(squeue -h -j 217336 -o %T | sort -u)" = PENDING
test "$(squeue -h -j 217337 -o %T | sort -u)" = PENDING
cp jobs.tsv jobs-before-node-recovery.tsv
scancel 217336 217337
printf 'prepare\\t217334\\noriginal-validation\\t217335\\n' > jobs.tsv
check=$(sbatch --parsable --array=2 --no-requeue --nodelist=comp002 --output=logs/recovery-check-%A_%a.out --error=logs/recovery-check-%A_%a.err validate.sh)
printf 'recovered-legacy-validation\\t%s\\n' "$check" >> jobs.tsv
eval=$(sbatch --parsable --array=0-23%6 --no-requeue --exclude=comp004,comp005,comp013,comp014,comp015,comp016 --dependency="afterok:$check" --kill-on-invalid-dep=yes --output=logs/recovery-eval-%A_%a.out --error=logs/recovery-eval-%A_%a.err evaluate.sh)
printf 'evaluation\\t%s\\n' "$eval" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$eval" --kill-on-invalid-dep=yes --output=logs/recovery-report-%j.out --error=logs/recovery-report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
        assert execute(client,command)==0
        with client.open_sftp() as sftp:
            for name in ['jobs.tsv','jobs-before-node-recovery.tsv']:
                sftp.get(REMOTE+'/'+name,str(LOCAL/name))
    finally: client.close()


if __name__=='__main__': main()
