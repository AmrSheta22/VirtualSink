"""Isolate the latest atomic checkpoint and submit a paired Qwen speed test."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
NAME = 'qwen05-checkpoint-speed-20261016-r1'
LOCAL = ROOT/'artifacts'/NAME
REMOTE = '/cluster/users/alex116u1/data/'+NAME
TRAINING = '/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1'
RUNTIME = '/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh'


def main():
    LOCAL.mkdir(exist_ok=False)
    paths = list((ROOT/'src').rglob('*.py'))+[ROOT/'hpc/benchmark_qwen_checkpoint_speed.py',Path(__file__)]
    files = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in paths}
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    production = json.loads((ROOT/'artifacts/qwen05-long-uptraining-20261014-r1/source-sha256.json').read_text())
    for name in ('native.py','recurrent_sink.py','train_recurrent.py','qwen05_long.py','recurrent_data.py'):
        name = 'src/virtualsink/uptraining/'+name
        assert production[name] == hashes[name], name
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(LOCAL/'release.tar.gz','w:gz') as tar:
        for name, data in files.items():
            entry=tarfile.TarInfo(name)
            entry.size=len(data)
            tar.addfile(entry,io.BytesIO(data))
    script=f'''#!/bin/bash
#SBATCH --job-name=qwen05-checkpoint-speed
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=comp002
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
status 'RUNNING paired Qwen checkpoint benchmark'
scontrol show job -d "$SLURM_JOB_ID" > {REMOTE}/slurm-allocation.txt
nvidia-smi --query-gpu=uuid,name,clocks.sm,temperature.gpu --format=csv > {REMOTE}/gpu-before.csv
python -u hpc/benchmark_qwen_checkpoint_speed.py --root {REMOTE} --training-root {TRAINING}
nvidia-smi --query-gpu=uuid,name,clocks.sm,temperature.gpu --format=csv > {REMOTE}/gpu-after.csv
'''
    (LOCAL/'benchmark.sh').write_bytes(script.encode())
    client=connect()
    try:
        if execute(client,f'test ! -e {REMOTE} && mkdir -p {REMOTE}/release/hpc {REMOTE}/logs'):
            raise RuntimeError('isolation failed')
        with client.open_sftp() as sftp:
            for name in ('release.tar.gz','source-sha256.json','benchmark.sh'):
                sftp.put(str(LOCAL/name),REMOTE+'/'+name)
            # Trainer saves a new temporary inode and atomically replaces last.pt.
            # A hard link pins the current immutable inode without copying 6 GB.
            if execute(client,f'tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release && cp {RUNTIME} {REMOTE}/release/hpc/hpc_runtime.sh && bash -n {REMOTE}/benchmark.sh && ln {TRAINING}/runs/ordinary/last.pt {REMOTE}/snapshot.pt'):
                raise RuntimeError('release/snapshot setup failed')
            st=sftp.stat(REMOTE+'/snapshot.pt')
            snapshot=dict(source=TRAINING+'/runs/ordinary/last.pt',remote_snapshot=REMOTE+'/snapshot.pt',bytes=st.st_size,mtime=st.st_mtime,
                semantics='hard link to immutable checkpoint inode; production uses atomic replacement')
            (LOCAL/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
            if execute(client,f'cd {REMOTE} && test ! -e jobs.tsv && job=$(sbatch --parsable --output=logs/benchmark-%j.out --error=logs/benchmark-%j.err benchmark.sh) && printf "benchmark\\t%s\\n" "$job" > jobs.tsv && cat jobs.tsv'):
                raise RuntimeError('inspect jobs.tsv before retrying')
            sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
    finally:
        client.close()


if __name__ == '__main__':
    main()
