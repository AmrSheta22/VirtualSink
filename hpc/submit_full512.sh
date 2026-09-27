#!/bin/bash
set -euo pipefail
export SMOL_PRETRAIN_ROOT="$HOME/data/smollm-full512-20260913"
export SMOL_PRETRAIN_STEPS=30
export SMOL_WINDOW_SIZE=0
export SMOL_SEQUENCE_LENGTH=512
export SMOL_EVAL_BLOCKS=64
export SMOL_TOKENIZER_REVISION=4ce2318a3628e77279c939ed6a9f3f03034402de
root="$SMOL_PRETRAIN_ROOT"
test ! -s "$root/jobs.tsv" || { cat "$root/jobs.tsv"; exit 2; }
mkdir -p "$root/logs" "$root/comparison"
cd "$root/release"
for f in hpc/*.sh; do bash -n "$f"; done
smoke=$(sbatch --parsable --output="$root/logs/smoke-%j.out" --error="$root/logs/smoke-%j.err" hpc/benchmark_full512.sh)
printf 'smoke\t%s\n' "$smoke" >> "$root/jobs.tsv"
prep=$(sbatch --parsable --output="$root/logs/prepare-%j.out" --error="$root/logs/prepare-%j.err" hpc/prepare_smollm_pretraining.sh)
printf 'prepare\t%s\n' "$prep" >> "$root/jobs.tsv"
std=$(sbatch --parsable --dependency="afterok:$smoke:$prep" --kill-on-invalid-dep=yes --job-name=full512_base --output="$root/logs/standard-%j.out" --error="$root/logs/standard-%j.err" hpc/train_smollm_pretraining.sh standard)
printf 'standard\t%s\n' "$std" >> "$root/jobs.tsv"
sink=$(sbatch --parsable --dependency="afterok:$smoke:$prep" --kill-on-invalid-dep=yes --job-name=full512_sink --output="$root/logs/query_sink-%j.out" --error="$root/logs/query_sink-%j.err" hpc/train_smollm_pretraining.sh query_sink)
printf 'query_sink\t%s\n' "$sink" >> "$root/jobs.tsv"
comparison=$(sbatch --parsable --dependency="afterok:$std:$sink" --kill-on-invalid-dep=yes --output="$root/logs/compare-%j.out" --error="$root/logs/compare-%j.err" hpc/compare_smollm_pretraining.sh)
printf 'compare\t%s\n' "$comparison" >> "$root/jobs.tsv"
diag=$(sbatch --parsable --dependency="afterok:$std:$sink:$comparison" --kill-on-invalid-dep=yes --job-name=full_stream_diag --output="$root/logs/full-diag-%j.out" --error="$root/logs/full-diag-%j.err" hpc/diagnose_streaming.sh "$root" "$root/diagnostics-full")
printf 'full-diagnostics\t%s\n' "$diag" >> "$root/jobs.tsv"
cat "$root/jobs.tsv"
