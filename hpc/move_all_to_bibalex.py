"""Checkpoint/stop MOHESR and add a fresh, matched seed-8 pair on Bibalex."""
import json
from pathlib import Path
import time

from bibalex_remote import connect as connect_bibalex, execute
from deploy_mohesr_full512 import connect, command, REMOTE
from report_split_full512 import BIB, LOCAL, ROOT


def main():
    mohesr = connect("hpclogin.mohesr.gov.eg", "alex005", "private_mohesr", "known_hosts_mohesr",
                     (ROOT / "server_access/password_mohesr.txt").read_text().strip())
    bibalex = connect_bibalex()
    try:
        assert execute(bibalex, f"test ! -d {BIB}/seeds/seed8; test ! -e {BIB}/jobs-seed8.tsv") == 0
        sftp = mohesr.open_sftp()
        run = REMOTE + "/seeds/seed8/runs/standard"
        before = sftp.stat(run + "/last.pt").st_mtime
        command(mohesr, "scancel --signal=USR1 --batch 11593")
        for _ in range(60):
            latest = command(mohesr, f"tail -n 1 {run}/metrics.jsonl").strip()
            row = json.loads(latest)
            if row.get("event") == "paused" and sftp.stat(run + "/last.pt").st_mtime > before:
                break
            time.sleep(2)
        else:
            raise RuntimeError("No new checkpointed pause confirmed; MOHESR was not cancelled")
        command(mohesr, f"set -e; ln {run}/last.pt {run}/migration-checkpoint.pt; scancel 11593; squeue -j 11593; ls -lh {run}/migration-checkpoint.pt")
        migration = dict(checkpoint=run + "/migration-checkpoint.pt", progress=row,
            reason="Fresh paired seed8 runs on Bibalex maintain one hardware/PyTorch runtime across all six arms.",
            old_job=11593, action="checkpoint preserved on MOHESR; job stopped; fresh seed8 pair on Bibalex")
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / "migration.json").write_text(json.dumps(migration, indent=2))
        target = bibalex.open_sftp()
        script = (ROOT / "hpc/train_bibalex_replication.sh").read_text().replace("${SPLIT_SEED_START:-9}", "${SPLIT_SEED_START:-8}")
        with target.open(BIB + "/release/hpc/train_bibalex_seed8.sh", "w") as handle:
            handle.write(script)
        submit = f'''set -euo pipefail
export SPLIT_EXPERIMENT={BIB}
cd {BIB}
test ! -e jobs-seed8.tsv
bash -n release/hpc/train_bibalex_seed8.sh
train=$(sbatch --parsable --array=0-1%2 --output=logs/seed8-%A_%a.out --error=logs/seed8-%A_%a.err release/hpc/train_bibalex_seed8.sh)
printf 'training-seed8-array\\t%s\\n' "$train" > jobs-seed8.tsv
cat jobs-seed8.tsv >> jobs.tsv
compare=$(sbatch --parsable --dependency="afterok:${{train}}_0:${{train}}_1" --kill-on-invalid-dep=yes --output=logs/compare-8-%j.out --error=logs/compare-8-%j.err compare.sh 8)
printf 'compare-seed8\\t%s\\n' "$compare" >> jobs-seed8.tsv
printf 'compare-seed8\\t%s\\n' "$compare" >> jobs.tsv
cat jobs-seed8.tsv
'''
        with target.open(BIB + "/submit-seed8.sh", "w") as handle:
            handle.write(submit)
        assert execute(bibalex, f"bash {BIB}/submit-seed8.sh") == 0
        target.get(BIB + "/jobs.tsv", str(LOCAL / "jobs-bibalex.tsv"))
        assignment = dict(mohesr=dict(root=REMOTE, job=11593, seeds=[], state="stopped", checkpoint=migration["checkpoint"]),
            bibalex=dict(root=BIB, seeds=[8, 9, 10]), microbatch=16,
            source="identical frozen numerical source", runtime_note="All final seed pairs use Bibalex K80/PyTorch1.12.1; abandoned MOHESR partial run excluded.")
        (LOCAL / "assignment.json").write_text(json.dumps(assignment, indent=2))
        for remote, path in ((sftp, REMOTE), (target, BIB)):
            with remote.open(path + "/split-assignment.json", "w") as handle:
                handle.write(json.dumps(assignment, indent=2))
    finally:
        mohesr.close()
        bibalex.close()


if __name__ == "__main__":
    main()
