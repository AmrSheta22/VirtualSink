"""Submit exactly two isolated benchmark jobs, sequentially on comp002."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
NAME = 'recurrent-speed-20261016-r1'
LOCAL = ROOT/'artifacts'/NAME
REMOTE = '/cluster/users/alex116u1/data/'+NAME
RUNTIME = '/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh'


def main():
    LOCAL.mkdir(exist_ok=False)
    paths = list((ROOT/'src').rglob('*.py')) + [ROOT/'hpc/benchmark_recurrent_speed.py', Path(__file__),
        ROOT/'tests/test_recurrent_optimized.py', ROOT/'tests/test_recurrent_sink.py']
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    (LOCAL/'source-sha256.json').write_text(json.dumps({n: hashlib.sha256(b).hexdigest() for n,b in files.items()}, indent=2))
    with tarfile.open(LOCAL/'release.tar.gz', 'w:gz') as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    for arm in ('current', 'optimized'):
        script = f'''#!/bin/bash
#SBATCH --job-name=recurrent-speed-{arm}
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=comp002
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
status 'RUNNING {arm} benchmark'
python -m pytest -q tests/test_recurrent_optimized.py
python -u hpc/benchmark_recurrent_speed.py --arm {arm} --output {REMOTE}/{arm}.json
'''
        (LOCAL/(arm+'.sh')).write_bytes(script.encode())
    client = connect()
    try:
        if execute(client, f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release/hpc {REMOTE}/logs'):
            raise RuntimeError('remote root already exists or creation failed')
        with client.open_sftp() as sftp:
            for name in ('release.tar.gz','source-sha256.json','current.sh','optimized.sh'):
                sftp.put(str(LOCAL/name), REMOTE+'/'+name)
            if execute(client, f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {RUNTIME} {REMOTE}/release/hpc/hpc_runtime.sh && bash -n {REMOTE}/current.sh && bash -n {REMOTE}/optimized.sh'):
                raise RuntimeError('release setup failed')
            command = f'''set -e
cd {REMOTE}
test ! -e jobs.tsv
current=$(sbatch --parsable --output=logs/current-%j.out --error=logs/current-%j.err current.sh)
printf 'current\\t%s\\n' "$current" > jobs.tsv
optimized=$(sbatch --parsable --dependency=afterok:$current --kill-on-invalid-dep=yes --output=logs/optimized-%j.out --error=logs/optimized-%j.err optimized.sh)
printf 'optimized\\t%s\\n' "$optimized" >> jobs.tsv
cat jobs.tsv
'''
            if execute(client, command):
                raise RuntimeError('inspect jobs.tsv before retrying')
            sftp.get(REMOTE+'/jobs.tsv', str(LOCAL/'jobs.tsv'))
    finally:
        client.close()


if __name__ == '__main__':
    main()
