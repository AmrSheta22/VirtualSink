"""Submit an isolated 70-update, 10-step warmup experiment from frozen source."""
import hashlib
import json
from pathlib import Path
import tarfile
from bibalex_remote import connect, execute

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/smollm-warm70-20261014"
REMOTE = "/cluster/users/alex116u1/data/smollm-warm70-20261014"
OLD = "/cluster/users/alex116u1/data/smollm-full512-20260913"


def main():
    stage = LOCAL / "release"
    assert stage.is_dir(), "Stage the frozen 20261009 source with recipe/train/test warmup changes first"
    tarpath = LOCAL / "release.tar.gz"
    hashes = {p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in stage.rglob("*.py") if "__pycache__" not in p.parts}
    (LOCAL / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
    with tarfile.open(tarpath, "w:gz") as archive:
        for p in stage.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".sh") and "__pycache__" not in p.parts:
                archive.add(p, arcname=p.relative_to(stage).as_posix())
    client = connect()
    try:
        assert execute(client, f"set -e; test ! -e {REMOTE}; mkdir -p {REMOTE}/release {REMOTE}/logs") == 0
        sftp = client.open_sftp()
        sftp.put(str(tarpath), REMOTE + "/release.tar.gz")
        sftp.put(str(LOCAL / "source-sha256.json"), REMOTE + "/source-sha256.json")
        assert execute(client, f"tar -xzf {REMOTE}/release.tar.gz -C {REMOTE}/release") == 0
        runtime = '''#!/bin/bash
: "${SLURM_JOB_ID:?Slurm required}"
source "$HOME/data/VirtualSink/venv/bin/activate"
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$WARM70_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
status() { echo "$(date -Is) $* job=$SLURM_JOB_ID host=$(hostname)"; }
status ENV_READY
'''
        prep = f'''#!/bin/bash
#SBATCH --job-name=smol70_data
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
cd "$WARM70_ROOT/release"
source hpc/hpc_runtime.sh
export HF_HOME={OLD}/hf-cache
python -m pytest -q tests/test_pretraining.py tests/test_native_uptraining.py
python -u -m virtualsink.pretraining.data --output-dir "$WARM70_ROOT/data" --steps 70 \\
 --sequence-length 512 --tokenizer-revision 4ce2318a3628e77279c939ed6a9f3f03034402de --allow-stackoverflow-substitution
python -u "$WARM70_ROOT/verify_data.py"
'''
        verify = f'''import hashlib,json,os
from pathlib import Path
root=Path(os.environ['WARM70_ROOT'])/'data'
new=json.loads((root/'manifest.json').read_text())
old=json.loads(Path('{OLD}/data/manifest.json').read_text())
assert new['steps']==70 and new['sequence_length']==512
for name,source in new['sources'].items():
 for split,spec in source['splits'].items():
  assert hashlib.sha256((root/spec['file']).read_bytes()).hexdigest()==spec['sha256']
  if split in ('validation','test'):
   assert spec['sha256']==old['sources'][name]['splits'][split]['sha256'], (name,split)
for split,spec in new['splits'].items():
 assert hashlib.sha256((root/(split+'-order.npy')).read_bytes()).hexdigest()==spec['order_sha256']
 if split in ('validation','test'): assert spec['order_sha256']==old['splits'][split]['order_sha256']
print('70-UPDATE DATA VERIFIED; ORIGINAL HELDOUT DATA IDENTICAL',flush=True)
'''
        train = '''#!/bin/bash
#SBATCH --job-name=smol70_warm10
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
cd "$WARM70_ROOT/release"
source hpc/hpc_runtime.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
index=${SLURM_ARRAY_TASK_ID:?}
seed=$((11 + index / 2))
variant=standard
if (( index % 2 )); then variant=query_sink; fi
status "RUNNING seed=$seed variant=$variant steps=70 warmup=10"
trainer_pid=
trap 'test -z "$trainer_pid" || kill -USR1 "$trainer_pid" 2>/dev/null || true' USR1
python -u -m virtualsink.pretraining.train --data-dir "$WARM70_ROOT/data" \\
 --output-dir "$WARM70_ROOT/seeds/seed$seed/runs/$variant" --variant "$variant" \\
 --model-seed "$seed" --steps 70 --warmup-steps 10 --microbatch 16 --chunk-size 32 \\
 --sequence-length 512 --window-size 0 --sink-weight-std .02 --sink-bias 4 \\
 --eval-every 1 --eval-blocks 64 --save-seconds 600 --max-hours 23 &
trainer_pid=$!
set +e
while true; do
 wait "$trainer_pid"; code=$?
 if kill -0 "$trainer_pid" 2>/dev/null; then continue; fi
 break
done
set -e
if [ "$code" -eq 75 ]; then
 status "CHECKPOINTED; requeue job=$SLURM_JOB_ID"
 scontrol requeue "$SLURM_JOB_ID"
 exit 0
fi
exit "$code"
'''
        compare = '''#!/bin/bash
#SBATCH --job-name=smol70_compare
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$WARM70_ROOT/release"
source hpc/hpc_runtime.sh
python -m virtualsink.pretraining.compare --root "$WARM70_ROOT/seeds/seed$1"
'''
        reporter = (ROOT / "hpc/report_mohesr_full512.py").read_text().replace(
            "for seed in (8, 9, 10):", "for seed in (11, 12, 13):")
        for name, data in {"release/hpc/hpc_runtime.sh":runtime,"prepare.sh":prep,
                           "verify_data.py":verify,"train.sh":train,"compare.sh":compare,
                           "report.py":reporter}.items():
            with sftp.open(REMOTE + "/" + name, "w") as handle:
                handle.write(data)
        protocol = dict(seeds=[11,12,13], arms=["standard","query_sink"], steps=70,
            warmup_steps=10, max_lr=.003, context=512, global_batch=2048, microbatch=16,
            source="Frozen completed full512 release with only explicit-warmup trainer/recipe changes",
            total_input_tokens_per_arm=70*1048576, heldout="identical original validation/test hashes required",
            note="Both warmup and training budget change; three new initialization seeds. Not a factorial isolation of warmup effect.")
        (LOCAL / "protocol.json").write_text(json.dumps(protocol, indent=2))
        sftp.put(str(LOCAL / "protocol.json"), REMOTE + "/protocol.json")
        submit = f'''set -euo pipefail
export WARM70_ROOT={REMOTE}
cd {REMOTE}
test ! -e jobs.tsv
for f in prepare.sh train.sh compare.sh; do bash -n "$f"; done
prep=$(sbatch --parsable --output=logs/prepare-%j.out --error=logs/prepare-%j.err prepare.sh)
printf 'prepare\\t%s\\n' "$prep" > jobs.tsv
train=$(sbatch --parsable --array=0-5%6 --dependency="afterok:$prep" --kill-on-invalid-dep=yes --output=logs/train-%A_%a.out --error=logs/train-%A_%a.err train.sh)
printf 'training-array\\t%s\\n' "$train" >> jobs.tsv
for seed in 11 12 13; do
 first=$(( (seed - 11) * 2 )); second=$((first + 1))
 cmp=$(sbatch --parsable --dependency="afterok:${{train}}_${{first}}:${{train}}_${{second}}" --kill-on-invalid-dep=yes --output=logs/compare-$seed-%j.out --error=logs/compare-$seed-%j.err compare.sh "$seed")
 printf 'compare-seed%s\\t%s\\n' "$seed" "$cmp" >> jobs.tsv
done
cat jobs.tsv
'''
        with sftp.open(REMOTE + "/submit.sh", "w") as handle:
            handle.write(submit)
        assert execute(client, f"bash {REMOTE}/submit.sh") == 0
        sftp.get(REMOTE + "/jobs.tsv", str(LOCAL / "jobs.tsv"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
