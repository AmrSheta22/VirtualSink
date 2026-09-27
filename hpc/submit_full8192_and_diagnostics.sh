#!/bin/bash
set -euo pipefail
export SMOL_PRETRAIN_ROOT="$HOME/data/smollm-full8192-20260913"
export SMOL_PRETRAIN_STEPS=30
export SMOL_WINDOW_SIZE=0
root="$SMOL_PRETRAIN_ROOT"
test ! -s "$root/jobs.tsv" || { cat "$root/jobs.tsv"; exit 2; }
mkdir -p "$root/logs" "$root/comparison"
cd "$root/release"
for f in hpc/*.sh; do bash -n "$f"; done
smoke=$(sbatch --parsable --output="$root/logs/smoke-%j.out" --error="$root/logs/smoke-%j.err" hpc/smoke_full8192.sh)
printf 'smoke\t%s\n' "$smoke" >> "$root/jobs.tsv"
swa=$(sbatch --parsable --dependency="afterok:214387:214388:$smoke" --kill-on-invalid-dep=yes --job-name=swa_stream_diag --output="$root/logs/swa-diag-%j.out" --error="$root/logs/swa-diag-%j.err" hpc/diagnose_streaming.sh "$HOME/data/smollm-swa8192-20260911" "$root/diagnostics-swa")
printf 'swa-diagnostics\t%s\n' "$swa" >> "$root/jobs.tsv"
std=$(sbatch --parsable --dependency="afterok:$smoke" --kill-on-invalid-dep=yes --job-name=full8192_base --output="$root/logs/standard-%j.out" --error="$root/logs/standard-%j.err" hpc/train_smollm_pretraining.sh standard)
printf 'standard\t%s\n' "$std" >> "$root/jobs.tsv"
sink=$(sbatch --parsable --dependency="afterok:$smoke" --kill-on-invalid-dep=yes --job-name=full8192_sink --output="$root/logs/query_sink-%j.out" --error="$root/logs/query_sink-%j.err" hpc/train_smollm_pretraining.sh query_sink)
printf 'query_sink\t%s\n' "$sink" >> "$root/jobs.tsv"
comparison=$(sbatch --parsable --dependency="afterok:$std:$sink" --kill-on-invalid-dep=yes --output="$root/logs/compare-%j.out" --error="$root/logs/compare-%j.err" hpc/compare_smollm_pretraining.sh)
printf 'compare\t%s\n' "$comparison" >> "$root/jobs.tsv"
diag=$(sbatch --parsable --dependency="afterok:$std:$sink:$comparison" --kill-on-invalid-dep=yes --job-name=full_stream_diag --output="$root/logs/full-diag-%j.out" --error="$root/logs/full-diag-%j.err" hpc/diagnose_streaming.sh "$root" "$root/diagnostics-full")
printf 'full-diagnostics\t%s\n' "$diag" >> "$root/jobs.tsv"
cat "$root/jobs.tsv"
