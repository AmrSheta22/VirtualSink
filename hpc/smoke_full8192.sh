#!/bin/bash
#SBATCH --job-name=smol_full_smoke
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp013,comp016
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
source "$HOME/data/phase3-k80-20260909/environment/bin/activate"
export PYTHONPATH="$SMOL_PRETRAIN_ROOT/release/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python -m pytest -q "$SMOL_PRETRAIN_ROOT/release/tests/test_streaming.py" "$SMOL_PRETRAIN_ROOT/release/tests/test_pretraining.py"
python -u - <<'PY'
import gc, torch
from virtualsink.pretraining.recipe import initialize,MODEL_CONFIG,optimizer_groups
from virtualsink.pretraining.train import language_loss
from virtualsink.uptraining.qwen import disable_sinks
for variant in ('standard','query_sink'):
    model=initialize(8,32,dict(MODEL_CONFIG,max_position_embeddings=8192),sink_weight_std=.02,sink_bias=4.).cuda()
    disable_sinks(model,variant=='standard')
    model.gradient_checkpointing_enable()
    opt=torch.optim.AdamW(optimizer_groups(model,variant),lr=.003,betas=(.9,.95),foreach=False)
    tokens=torch.randint(0,49152,(1,8193),device='cuda')
    loss,_=language_loss(model,tokens,32)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step(); torch.cuda.synchronize()
    print(variant,'8192 FULL ATTENTION FORWARD/BACKWARD/ADAM PASSED',loss.item(),torch.cuda.max_memory_allocated(),flush=True)
    del model,opt,tokens,loss,_
    gc.collect(); torch.cuda.empty_cache()
PY
