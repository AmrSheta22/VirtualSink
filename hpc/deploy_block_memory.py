"""Freeze block-memory code, submit a GPU gate and two dependent small pilots."""
import argparse
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
DATA = '/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1/data'
RUNTIME = '/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011-r2/release/hpc/hpc_runtime.sh'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError('attempt must be positive')
    name = f'block-memory-qwen05-20261018-r{args.attempt}'
    local = ROOT / 'artifacts' / name
    remote = '/cluster/users/alex116u1/data/' + name
    local.mkdir(exist_ok=False)
    paths = list((ROOT / 'src').rglob('*.py')) + [ROOT / 'tests/test_block_memory.py',
        ROOT / 'tests/test_recurrent_sink.py', ROOT / 'hpc/run_block_memory.py',
        ROOT / 'hpc/collect_block_memory.py', Path(__file__)]
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(paths)}
    client = connect()
    try:
        with client.open_sftp() as sftp:
            with sftp.open(RUNTIME, 'rb') as source:
                files['hpc/hpc_runtime.sh'] = source.read()
            with sftp.open(DATA + '/manifest.json', 'rb') as source:
                data_manifest = source.read()
            (local / 'data-manifest.json').write_bytes(data_manifest)
            hashes = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
            (local / 'source-sha256.json').write_text(json.dumps(hashes, indent=2))
            with tarfile.open(local / 'release.tar.gz', 'w:gz') as archive:
                for filename, content in files.items():
                    entry = tarfile.TarInfo(filename); entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
            plan = dict(name=name, remote=remote, model='Qwen/Qwen2.5-0.5B',
                revision='060db6499f32faf8b98477b0a26969ef7d8b9987',
                data=DATA, data_manifest_sha256=hashlib.sha256(data_manifest).hexdigest(),
                gate=dict(capacity=16, tokens=65, optimizer_steps=2,
                          checks='numerical tests, full pretrained streaming parity, all memory gradients and updates'),
                pilot=dict(arms=['block', 'null'], seed=1729, steps=128, max_input_tokens=65536,
                    capacity=128, block_size=128, segment=256, unroll=2,
                    mode='memory only; pretrained backbone frozen', learning_rate=.01,
                    initialization='role and null biases -8, random learned pool query; no calibration',
                    validation='one longest full article; no model selection',
                    test='two longest full articles, selected by length only',
                    interventions=['enabled', 'fifo', 'no_memory', 'zero_values', 'mean_only', 'importance_only', 'free_only'],
                    checkpoint='fixed final step; save every 16 steps'),
                limitations=['one seed and small budget', 'growing summary bank',
                    'matched physical window, unequal total memory', 'TBPTT gradient horizon 512 tokens',
                    'pilot budget tests viability, not a definitive architecture comparison'])
            (local / 'plan.json').write_text(json.dumps(plan, indent=2))
            common = f'''#!/bin/bash
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
'''
            runtime = f'''set -euo pipefail
cd {remote}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={remote}/release/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
'''
            command = f'python -u hpc/run_block_memory.py --root {remote} --data {DATA}'
            scripts = {
                'gate.sh': common + '#SBATCH --job-name=block_memory_gate\n#SBATCH --time=00:45:00\n' + runtime +
                    'status "RUNNING block-memory correctness and Qwen gate"\n' +
                    'python -m pytest -q tests/test_block_memory.py tests/test_recurrent_sink.py\n' + command + ' --arm gate\n',
                'pilot.sh': common + '#SBATCH --job-name=block_memory_pilot\n#SBATCH --time=12:00:00\n' + runtime +
                    'status "RUNNING block-memory pilot $ARM"\n' + command + ' --arm "$ARM" --steps 128\n'}
            for filename, content in scripts.items():
                (local / filename).write_bytes(content.encode('utf-8'))
            q = shlex.quote(remote)
            if execute(client, f'set -e; test ! -e {q}; mkdir -p {q}/release {q}/logs'):
                raise RuntimeError('remote isolation failed')
            for filename in ['release.tar.gz', 'source-sha256.json', 'plan.json', *scripts]:
                sftp.put(str(local / filename), remote + '/' + filename)
            if execute(client, f'set -e; tar -xzf {q}/release.tar.gz -C {q}/release; bash -n {q}/gate.sh; bash -n {q}/pilot.sh'):
                raise RuntimeError('script validation failed')
            submission = f'''set -e
cd {remote}
test ! -e jobs.tsv
gate=$(sbatch --parsable --output=logs/gate-%j.out --error=logs/gate-%j.err gate.sh)
printf 'gate\\t%s\\n' "$gate" > jobs.tsv
for arm in block null; do
job=$(sbatch --parsable --dependency="afterok:$gate" --kill-on-invalid-dep=yes --export="ALL,ARM=$arm" --output="logs/$arm-%j.out" --error="logs/$arm-%j.err" pilot.sh)
printf '%s\\t%s\\n' "$arm" "$job" >> jobs.tsv
done
cat jobs.tsv
'''
            if execute(client, submission):
                raise RuntimeError('submission failed; inspect remote jobs.tsv before retrying')
            sftp.get(remote + '/jobs.tsv', str(local / 'jobs.tsv'))
    finally:
        client.close()


if __name__ == '__main__':
    main()
