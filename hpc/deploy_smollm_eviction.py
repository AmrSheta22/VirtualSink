"""Submit an isolated read-only eviction evaluation of all six completed pairs."""
import hashlib
import json
from pathlib import Path
import tarfile

from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/cluster/users/alex116u1/data/smollm-eviction-20261016'
LOCAL = ROOT / 'artifacts/smollm-eviction-20261016'
TRAINING = '/cluster/users/alex116u1/data/smollm-warm70-20261014'


def main():
    LOCAL.mkdir(exist_ok=False)
    files = [ROOT / 'hpc' / name for name in ['evaluate_smollm_eviction.py',
             'smollm_streaming_decoder.py', 'smollm_rebased_reference.py']]
    files += [ROOT / 'tests/test_smollm_eviction.py']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (LOCAL / 'source-sha256.json').write_text(json.dumps(hashes, indent=2))
    protocol = dict(seeds=list(range(8,14)), capacity=512, predictions_per_source=4096,
        sources=['fineweb','cosmopedia','python','math','stackoverflow'],
        policies_per_seed_source=12, total_predictions=1474560,
        physical_budget='512 entries including current token; trim before attention',
        selection='First4097 tokens of each original held-out source stream; SHA256 matched across experiments',
        regions=dict(pre=[0,512], post=[512,4096], late=[2048,4096]),
        controls='Two standard prefix policies and four enabled/disabled virtual policies, each under absolute and cache-relative RoPE',
        endpoint='Within-checkpoint prefix-retention rescue and sink-enabling benefit after eviction',
        interpretation='Replacement supported only if prefix retention helps without sink but its benefit is reduced with sink enabled, while enabled FIFO remains competitive with enabled prefix retention. No rescue means uninformative, not successful replacement.',
        uncertainty='Descriptive fixed-source screen; equal source weighting, per-seed reporting; no document confidence interval or pooled30/70update estimate',
        caveats='EOS-packed streams cross documents; not a long-book StreamingLLM replication. Relative RoPE also changes query-conditioned sink scores. No training or checkpoint changes.')
    (LOCAL / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    with tarfile.open(LOCAL / 'release.tar.gz', 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    client = connect()
    try:
        assert execute(client, f'set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release/hpc {REMOTE}/results {REMOTE}/validation {REMOTE}/logs') == 0
        sftp = client.open_sftp()
        for name in ['release.tar.gz','source-sha256.json','protocol.json']:
            sftp.put(str(LOCAL / name), REMOTE + '/' + name)
        assert execute(client, f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {TRAINING}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/') == 0
        common = f'''set -euo pipefail
cd {REMOTE}/release
export WARM70_ROOT={TRAINING}
source hpc/hpc_runtime.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export FROZEN_UPTRAINING_ROOT={TRAINING}/release/src/virtualsink/uptraining
'''
        cpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=cpu\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G\n#SBATCH --time=00:30:00\n'
        gpu = '#SBATCH --account=g.alex116\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --exclude=comp013,comp016\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=24G\n'
        call = f'python -u hpc/evaluate_smollm_eviction.py'
        scripts = {
            'prepare.sh': '#!/bin/bash\n#SBATCH --job-name=smol_evict_pin\n' + cpu + common + f'{call} prepare --root {REMOTE}\n',
            'validate.sh': '#!/bin/bash\n#SBATCH --job-name=smol_evict_check\n#SBATCH --time=02:00:00\n' + gpu + common + f'python -m pytest -q tests/test_smollm_eviction.py\n{call} validate --root {REMOTE}\n',
            'evaluate.sh': '#!/bin/bash\n#SBATCH --job-name=smol_evict_eval\n#SBATCH --time=12:00:00\n' + gpu + common + f'{call} evaluate --root {REMOTE}\n',
            'report.sh': '#!/bin/bash\n#SBATCH --job-name=smol_evict_report\n' + cpu + common + f'{call} summarize --root {REMOTE}\n',
        }
        scripts['submit.sh'] = f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
test ! -e jobs.tsv
for script in prepare.sh validate.sh evaluate.sh report.sh; do bash -n "$script"; done
pin=$(sbatch --parsable --output=logs/pin-%j.out --error=logs/pin-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$pin" > jobs.tsv
check=$(sbatch --parsable --array=0-5%6 --dependency="afterok:$pin" --kill-on-invalid-dep=yes --output=logs/check-%A_%a.out --error=logs/check-%A_%a.err validate.sh)
printf 'validation\\t%s\\n' "$check" >> jobs.tsv
eval=$(sbatch --parsable --array=0-29%6 --dependency="afterok:$check" --kill-on-invalid-dep=yes --output=logs/eval-%A_%a.out --error=logs/eval-%A_%a.err evaluate.sh)
printf 'evaluation\\t%s\\n' "$eval" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$eval" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
        for name, content in scripts.items():
            (LOCAL / name).write_bytes(content.encode())
            with sftp.open(REMOTE + '/' + name, 'w') as handle:
                handle.write(content)
        assert execute(client, f'bash {REMOTE}/submit.sh') == 0
        sftp.get(REMOTE + '/jobs.tsv', str(LOCAL / 'jobs.tsv'))
    finally:
        client.close()


if __name__ == '__main__':
    main()
