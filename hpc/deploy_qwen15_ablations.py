"""Freeze two authorized sink experiments and dependent twelve-article streaming tests."""
import hashlib
import argparse
import json
from pathlib import Path
import tarfile
from bibalex_remote import connect,execute

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'artifacts/qwen15-sink-ablations-20261011'
REMOTE = '/cluster/users/alex116u1/data/qwen15-sink-ablations-20261011'
ORIGINAL = '/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010'


def scripts():
    common = f'''set -euo pipefail
cd {REMOTE}/release
source hpc/hpc_runtime.sh
export PYTHONPATH={REMOTE}/release/src
export QWEN_SEEDS_ROOT={REMOTE}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
'''
    gpu = '''#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
'''
    cpu = '''#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
'''
    tests = ('tests/test_multi_sink.py tests/test_sink_schedule.py tests/test_swa_uptraining.py '
             'tests/test_native_uptraining.py tests/test_uptrained_streaming.py tests/test_qwen15_streaming.py '
             'tests/test_rebased_streaming.py tests/test_streaming_confirmation.py')
    smoke = '#!/bin/bash\n#SBATCH --job-name=q15_ablate_smoke\n#SBATCH --time=02:00:00\n'+gpu+common+f'''python -m pytest -q {tests}
python -u hpc/smoke_qwen15_ablations.py
'''
    training = (ROOT/'hpc/train_qwen15_seeds.sh').read_text()
    training = training.replace('#SBATCH --job-name=qwen15_seed','#SBATCH --job-name=q15_ablate')
    start = training.index('index="${SLURM_ARRAY_TASK_ID:?}"')
    end = training.index('if [ -e "$run/summary.json" ]')
    training = training[:start]+'''index="${RUN_INDEX:?}"
seed=1729
variant=query_sink
if [ "$index" = 0 ]; then
  label=highlr-single
  settings=(--sink-count 1 --base-lr 2e-5 --sink-lr 5e-4 --sink-match-step 300 --auxiliary anti_collapse --gamma .01)
elif [ "$index" = 1 ]; then
  label=highlr-four
  settings=(--sink-count 4 --sink-value-init-scale 1e-9 --base-lr 1e-4 --sink-lr 1e-4 --sink-match-step 0 --auxiliary none --gamma 0)
else exit 2; fi
run="$QWEN_SEEDS_ROOT/runs/$label"
'''+training[end:]
    training = training.replace('--auxiliary anti_collapse --gamma .01 --seed "$seed"', '"${settings[@]}" --seed "$seed"')
    pin = '#!/bin/bash\n#SBATCH --job-name=q15_ablate_pin\n#SBATCH --time=00:30:00\n'+cpu+common+f'python -u -m virtualsink.uptraining.qwen15_ablation_streaming --root {REMOTE} --pin\n'
    validate = '#!/bin/bash\n#SBATCH --job-name=q15_ablate_parity\n#SBATCH --time=02:00:00\n'+gpu+common+f'python -u -m virtualsink.uptraining.qwen15_ablation_streaming --root {REMOTE} --validate-only\n'
    articles = '#!/bin/bash\n#SBATCH --job-name=q15_ablate_stream\n#SBATCH --time=08:00:00\n'+gpu+common+f'python -u -m virtualsink.uptraining.qwen15_ablation_streaming --root {REMOTE} --article-index "${{SLURM_ARRAY_TASK_ID:?}}"\n'
    report = '#!/bin/bash\n#SBATCH --job-name=q15_ablate_report\n#SBATCH --time=00:30:00\n'+cpu+common+f'python -u -m virtualsink.uptraining.summarize_qwen15_ablations --root {REMOTE}\n'
    submit = f'''#!/bin/bash
set -euo pipefail
cd {REMOTE}
export QWEN_SEEDS_ROOT={REMOTE}
test ! -e jobs.tsv
for s in smoke.sh train.sh pin.sh validate.sh articles.sh report.sh; do bash -n "$s"; done
smoke=$(sbatch --parsable --output=logs/smoke-%j.out --error=logs/smoke-%j.err smoke.sh)
printf 'smoke\\t%s\\n' "$smoke" > jobs.tsv
single=$(sbatch --parsable --job-name=q15_single_aux --export=ALL,RUN_INDEX=0 --dependency="afterok:$smoke" --kill-on-invalid-dep=yes --output=logs/single-%j.out --error=logs/single-%j.err train.sh)
printf 'training-single-aux\\t%s\\n' "$single" >> jobs.tsv
four=$(sbatch --parsable --job-name=q15_four_equal --export=ALL,RUN_INDEX=1 --dependency="afterok:$smoke" --kill-on-invalid-dep=yes --output=logs/four-%j.out --error=logs/four-%j.err train.sh)
printf 'training-four-equal\\t%s\\n' "$four" >> jobs.tsv
pin=$(sbatch --parsable --dependency="afterok:$single:$four" --kill-on-invalid-dep=yes --output=logs/pin-%j.out --error=logs/pin-%j.err pin.sh)
printf 'streaming-pin\\t%s\\n' "$pin" >> jobs.tsv
validate=$(sbatch --parsable --dependency="afterok:$pin" --kill-on-invalid-dep=yes --output=logs/parity-%j.out --error=logs/parity-%j.err validate.sh)
printf 'streaming-validation\\t%s\\n' "$validate" >> jobs.tsv
array=$(sbatch --parsable --array=0-11%4 --dependency="afterok:$validate" --kill-on-invalid-dep=yes --output=logs/stream-%A_%a.out --error=logs/stream-%A_%a.err articles.sh)
printf 'streaming-array\\t%s\\n' "$array" >> jobs.tsv
report=$(sbatch --parsable --dependency="afterany:$array" --kill-on-invalid-dep=yes --output=logs/report-%j.out --error=logs/report-%j.err report.sh)
printf 'streaming-report\\t%s\\n' "$report" >> jobs.tsv
cat jobs.tsv
'''
    return {'smoke.sh':smoke,'train.sh':training,'pin.sh':pin,'validate.sh':validate,'articles.sh':articles,'report.sh':report,'submit.sh':submit}


def main():
    global LOCAL,REMOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',choices=('', 'r2'),default='')
    args = parser.parse_args()
    if args.attempt:
        LOCAL = ROOT/('artifacts/qwen15-sink-ablations-20261011-'+args.attempt)
        REMOTE += '-'+args.attempt
    LOCAL.mkdir(parents=True,exist_ok=True)
    plan = dict(remote_root=REMOTE,model='Qwen/Qwen2.5-1.5B',revision='8faed761d45a263340a0528343f099c05c9a4323',
        retry_of='qwen15-sink-ablations-20261011' if args.attempt else None,
        pretrained_source=ORIGINAL,seed=1729,steps=1000,seq_len=1024,window=512,batch_size=1,accumulation=8,
        input_tokens_per_arm=8192000,mode='attention',precision='fp32',warmup_steps=100,
        arms=[dict(label='highlr-single',sink_count=1,base_lr=2e-5,sink_lr=5e-4,sink_match_step=300,auxiliary='anti_collapse',gamma=.01),
              dict(label='highlr-four',sink_count=4,base_lr=1e-4,sink_lr=1e-4,sink_match_step=0,auxiliary='none',gamma=0,
                   sink_value_init_scale=1e-9,value_scales=[0,1e-9/3,2e-9/3,1e-9],first_value_fixed_zero=True,
                   three_value_vectors_trainable=True,value_scale_reference='pretrained V-projection weight RMS per layer',
                   initial_bias='-3.5-log(4), preserves aggregate initial virtual logit')],
        streaming=dict(articles=12,predictions_per_policy=8192,capacity=512,policies=7,concurrency=4,
                       dependency='afterok both training jobs, pin checkpoints, validate full-model parity'))
    (LOCAL/'plan.json').write_text(json.dumps(plan,indent=2))
    paths = sorted(p for folder in ['src','tests'] for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts)
    paths += [ROOT/'hpc/smoke_qwen15_ablations.py',Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (LOCAL/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    archive = LOCAL/'release.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in paths:
            t.add(p,arcname=p.relative_to(ROOT).as_posix())
    generated = scripts()
    for name,content in generated.items():
        (LOCAL/name).write_bytes(content.replace('\r\n','\n').encode())
    client = connect()
    try:
        assert execute(client,f'set -e; test ! -e {REMOTE}; test -e {ORIGINAL}/data/manifest.json; mkdir -p {REMOTE}/release/hpc {REMOTE}/logs {REMOTE}/results {REMOTE}/validation; ln -s {ORIGINAL}/data {REMOTE}/data')==0
        sftp = client.open_sftp()
        for p in [archive,LOCAL/'plan.json',LOCAL/'source-sha256.json']:
            sftp.put(str(p),REMOTE+'/'+p.name)
        assert execute(client,f'set -e; tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release; cp {ORIGINAL}/release/hpc/hpc_runtime.sh {REMOTE}/release/hpc/hpc_runtime.sh')==0
        for name in generated:
            sftp.put(str(LOCAL/name),REMOTE+'/'+name)
        assert execute(client,f'bash {REMOTE}/submit.sh')==0
        sftp.get(REMOTE+'/jobs.tsv',str(LOCAL/'jobs.tsv'))
        print('REMOTE_ROOT',REMOTE,flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
