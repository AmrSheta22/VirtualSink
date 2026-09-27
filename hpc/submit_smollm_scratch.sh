#!/bin/bash
set -euo pipefail

export SMOL_PRETRAIN_STEPS="${1:-30}"
export SMOL_TOKENIZER_REVISION="${2:-4ce2318a3628e77279c939ed6a9f3f03034402de}"
export SMOL_PRETRAIN_ROOT="$HOME/data/smollm-scratch-$(date +%Y%m%d)"

echo "=== SmolLM-135M From-Scratch Paired Experiment ==="
echo "Steps: $SMOL_PRETRAIN_STEPS"
echo "Experiment Root: $SMOL_PRETRAIN_ROOT"

if test -s "$SMOL_PRETRAIN_ROOT/jobs.tsv"; then
  echo "Submission journal already exists at $SMOL_PRETRAIN_ROOT/jobs.tsv; inspect it before resubmission."
  cat "$SMOL_PRETRAIN_ROOT/jobs.tsv"
  exit 2
fi

mkdir -p "$SMOL_PRETRAIN_ROOT/logs" "$SMOL_PRETRAIN_ROOT/comparison"

# Ensure release checkout exists and is at the latest commit
if ! test -d "$SMOL_PRETRAIN_ROOT/release"; then
  cd "$HOME/data/VirtualSink"
  if test -f "$HOME/data/smollm-scratch-20260910.bundle"; then
    git fetch "$HOME/data/smollm-scratch-20260910.bundle" refs/heads/codex/smollm-from-scratch
    git worktree add --detach "$SMOL_PRETRAIN_ROOT/release" FETCH_HEAD
  elif test -f "$HOME/data/smollm-scratch.bundle"; then
    git fetch "$HOME/data/smollm-scratch.bundle" refs/heads/codex/smollm-from-scratch
    git worktree add --detach "$SMOL_PRETRAIN_ROOT/release" FETCH_HEAD
  else
    git worktree add --detach "$SMOL_PRETRAIN_ROOT/release" HEAD
  fi
else
  if test -d "$SMOL_PRETRAIN_ROOT/release/.git" && test -f "$HOME/data/smollm-scratch-20260910.bundle"; then
    git -C "$SMOL_PRETRAIN_ROOT/release" fetch "$HOME/data/smollm-scratch-20260910.bundle" refs/heads/codex/smollm-from-scratch
    git -C "$SMOL_PRETRAIN_ROOT/release" reset --hard FETCH_HEAD
  fi
fi

cd "$SMOL_PRETRAIN_ROOT/release"

# Validate shell syntax
bash -n hpc/prepare_smollm_pretraining.sh
bash -n hpc/train_smollm_pretraining.sh
bash -n hpc/compare_smollm_pretraining.sh

# 1. CPU Data Preparation
prep_id=$(sbatch --parsable --chdir="$PWD" \
  --output="$SMOL_PRETRAIN_ROOT/logs/prepare-%j.out" \
  --error="$SMOL_PRETRAIN_ROOT/logs/prepare-%j.err" \
  --export=ALL,SMOL_PRETRAIN_ROOT="$SMOL_PRETRAIN_ROOT",SMOL_PRETRAIN_STEPS="$SMOL_PRETRAIN_STEPS",SMOL_TOKENIZER_REVISION="$SMOL_TOKENIZER_REVISION" \
  hpc/prepare_smollm_pretraining.sh)
printf 'prepare\t%s\n' "$prep_id" >> "$SMOL_PRETRAIN_ROOT/jobs.tsv"
echo "Submitted CPU data preparation: job $prep_id"

# 2. Standard Baseline GPU Training
std_id=$(sbatch --parsable --chdir="$PWD" \
  --dependency="afterok:$prep_id" --kill-on-invalid-dep=yes \
  --job-name="smol_standard" \
  --output="$SMOL_PRETRAIN_ROOT/logs/train-standard-%j.out" \
  --error="$SMOL_PRETRAIN_ROOT/logs/train-standard-%j.err" \
  --export=ALL,SMOL_PRETRAIN_ROOT="$SMOL_PRETRAIN_ROOT",SMOL_PRETRAIN_STEPS="$SMOL_PRETRAIN_STEPS" \
  hpc/train_smollm_pretraining.sh standard)
printf 'train-standard\t%s\n' "$std_id" >> "$SMOL_PRETRAIN_ROOT/jobs.tsv"
echo "Submitted standard baseline training: job $std_id (afterok:$prep_id)"

# 3. Query-Sink Virtual Sink GPU Training
sink_id=$(sbatch --parsable --chdir="$PWD" \
  --dependency="afterok:$prep_id" --kill-on-invalid-dep=yes \
  --job-name="smol_query_sink" \
  --output="$SMOL_PRETRAIN_ROOT/logs/train-query_sink-%j.out" \
  --error="$SMOL_PRETRAIN_ROOT/logs/train-query_sink-%j.err" \
  --export=ALL,SMOL_PRETRAIN_ROOT="$SMOL_PRETRAIN_ROOT",SMOL_PRETRAIN_STEPS="$SMOL_PRETRAIN_STEPS" \
  hpc/train_smollm_pretraining.sh query_sink)
printf 'train-query_sink\t%s\n' "$sink_id" >> "$SMOL_PRETRAIN_ROOT/jobs.tsv"
echo "Submitted query_sink training: job $sink_id (afterok:$prep_id)"

# 4. Result Comparison Job (CPU, runs directly after BOTH training jobs complete)
compare_id=$(sbatch --parsable --chdir="$PWD" \
  --dependency="afterok:$std_id:$sink_id" --kill-on-invalid-dep=yes \
  --job-name="smol_compare" \
  --output="$SMOL_PRETRAIN_ROOT/logs/compare-%j.out" \
  --error="$SMOL_PRETRAIN_ROOT/logs/compare-%j.err" \
  --export=ALL,SMOL_PRETRAIN_ROOT="$SMOL_PRETRAIN_ROOT",SMOL_PRETRAIN_STEPS="$SMOL_PRETRAIN_STEPS" \
  hpc/compare_smollm_pretraining.sh)
printf 'compare\t%s\n' "$compare_id" >> "$SMOL_PRETRAIN_ROOT/jobs.tsv"
echo "Submitted comparison job: job $compare_id (afterok:$std_id:$sink_id)"

echo ""
echo "=== All Jobs Submitted Successfully ==="
cat "$SMOL_PRETRAIN_ROOT/jobs.tsv"
echo ""
squeue -u alex116u1
