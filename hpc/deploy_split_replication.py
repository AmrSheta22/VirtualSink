"""Move seeds 9/10 to four Bibalex jobs and retain seed 8 on MOHESR."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

from bibalex_remote import connect as connect_bibalex, execute
from deploy_mohesr_full512 import connect, command, REMOTE

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/split-full512-20261009"
BIB = "/cluster/users/alex116u1/data/smollm-full512-replication-20261009"


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Numerical source must remain identical to MOHESR's frozen training code.
    frozen = json.loads((ROOT / "artifacts/mohesr-full512-20261009/source-sha256.json").read_text())
    for name, digest in frozen.items():
        if name.startswith("src/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    release = LOCAL / "release.tar.gz"
    with tarfile.open(release, "w:gz") as archive:
        for folder in ("src", "tests", "hpc"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".sh") and "__pycache__" not in path.parts:
                    archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    bibalex = connect_bibalex()
    mohesr = None
    try:
        if "--resume" in sys.argv:
            assert execute(bibalex, f"test -d {BIB}/release; test ! -e {BIB}/jobs.tsv") == 0
        else:
            assert execute(bibalex, f"set -e; test ! -e {BIB}; mkdir -p {BIB}/release {BIB}/logs; ln -s /cluster/users/alex116u1/data/smollm-full512-20260913/data {BIB}/data") == 0
        sftp = bibalex.open_sftp()
        sftp.put(str(release), BIB + "/release.tar.gz")
        assert execute(bibalex, f"tar -xzf {BIB}/release.tar.gz -C {BIB}/release") == 0
        runtime = '''#!/bin/bash
: "${SLURM_JOB_ID:?Slurm allocation required}"
source "$HOME/data/VirtualSink/venv/bin/activate"
# K80 requires the verified CUDA 11.3 / PyTorch 1.12 runtime.
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
status() { echo "$(date -Is) $* job=$SLURM_JOB_ID host=$(hostname)"; }
status ENV_READY
'''
        with sftp.open(BIB + "/release/hpc/hpc_runtime.sh", "w") as handle:
            handle.write(runtime)
        compare = '''#!/bin/bash
#SBATCH --job-name=smol512_compare
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SPLIT_EXPERIMENT/release"
source hpc/hpc_runtime.sh
export PYTHONPATH="$SPLIT_EXPERIMENT/release/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m virtualsink.pretraining.compare --root "$SPLIT_EXPERIMENT/seeds/seed$1"
'''
        with sftp.open(BIB + "/compare.sh", "w") as handle:
            handle.write(compare)
        # Narrow the MOHESR launcher before launching Bibalex to avoid duplicates.
        mohesr = connect("hpclogin.mohesr.gov.eg", "alex005", "private_mohesr", "known_hosts_mohesr",
                         (ROOT / "server_access/password_mohesr.txt").read_text().strip())
        target = mohesr.open_sftp()
        command(mohesr, f"test ! -d {REMOTE}/seeds/seed9/runs; test ! -d {REMOTE}/seeds/seed10/runs")
        for name in ("run_mohesr_full512.sh", "report_mohesr_full512.py"):
            target.put(str(ROOT / "hpc" / name), REMOTE + "/release/hpc/" + name)
        with target.open(REMOTE + "/arm-indices.txt", "w") as handle:
            handle.write("0 1\n")
        command(mohesr, f"bash -n {REMOTE}/release/hpc/run_mohesr_full512.sh; scancel --signal=USR1 --batch 11593")
        submit = f'''set -euo pipefail
export SPLIT_EXPERIMENT={BIB}
cd {BIB}
test ! -e jobs.tsv
bash -n release/hpc/train_bibalex_replication.sh
bash -n compare.sh
train=$(sbatch --parsable --array=0-3%4 --output=logs/train-%A_%a.out --error=logs/train-%A_%a.err release/hpc/train_bibalex_replication.sh)
printf 'training-array\\t%s\\n' "$train" > jobs.tsv
for seed in 9 10; do
  first=$(( (seed - 9) * 2 )); second=$((first + 1))
  compare=$(sbatch --parsable --dependency="afterok:${{train}}_${{first}}:${{train}}_${{second}}" --kill-on-invalid-dep=yes --output=logs/compare-$seed-%j.out --error=logs/compare-$seed-%j.err compare.sh "$seed")
  printf 'compare-seed%s\\t%s\\n' "$seed" "$compare" >> jobs.tsv
done
cat jobs.tsv
'''
        with sftp.open(BIB + "/submit.sh", "w") as handle:
            handle.write(submit)
        assert execute(bibalex, f"bash {BIB}/submit.sh") == 0
        sftp.get(BIB + "/jobs.tsv", str(LOCAL / "jobs-bibalex.tsv"))
        assignment = dict(mohesr=dict(root=REMOTE, job=11593, seeds=[8]),
                          bibalex=dict(root=BIB, seeds=[9, 10]),
                          microbatch=16, source="identical frozen numerical source",
                          runtime_note="Initialization seed is confounded with cluster/runtime across pairs; each pair is matched within one cluster.")
        (LOCAL / "assignment.json").write_text(json.dumps(assignment, indent=2))
        with target.open(REMOTE + "/split-assignment.json", "w") as handle:
            handle.write(json.dumps(assignment, indent=2))
        with sftp.open(BIB + "/split-assignment.json", "w") as handle:
            handle.write(json.dumps(assignment, indent=2))
        print("BIBALEX_ROOT", BIB)
    finally:
        bibalex.close()
        if mohesr:
            mohesr.close()


if __name__ == "__main__":
    main()
