#!/bin/bash
#SBATCH --job-name=full512_head_watch
#SBATCH --account=g.alex116
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
root="$HOME/data/smollm-full512-20260913"
set +e
python3 "$root/concentration/monitor_concentration.py" --root "$root" --reference "$HOME/data/smollm-swa8192-20260911/runs/query_sink/summary.json" --output "$root/concentration/results" --watch 60
code=$?
set -e
if [ "$code" -eq 75 ]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$code"
