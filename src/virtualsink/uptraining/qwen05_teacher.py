"""Slurm-only matched pretrained-Qwen full-block-teacher language-model screen."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .native import load_native
from .recurrent_sink import RecurrentSinkLM
from .recall_experiment import atomic_json
from .train_recurrent import token_losses

COMMON = dict(seed=1729, steps=64, warmup_steps=16, capacity=32, length=128,
              base_lr=2e-5, memory_lr=3e-4, initial_mass=1e-4,
              timescales=[128.,512.,2048.], teacher_weight=1., loss_chunk=8)


def objective(model, tokens, targets, teacher=False):
    output = model(tokens, memory_teacher=teacher)
    losses = token_losses(model, output.hidden, targets, COMMON['loss_chunk'])
    # Both auxiliary objectives score the same post-eviction targets.
    return losses[model.capacity:].mean()


def optimizer(model, adapting=False, previous=None):
    groups = model.parameter_groups('full' if adapting else 'memory', COMMON['base_lr'], COMMON['memory_lr'])
    result = torch.optim.AdamW(groups, foreach=False)
    if previous is not None:
        for group in result.param_groups:
            for parameter in group['params']:
                if parameter in previous.state:
                    result.state[parameter] = previous.state[parameter]
    return result


@torch.no_grad()
def evaluate(model, data, interventions=False):
    model.eval()
    flags = {'ordinary': {}, 'memory_disabled_fifo': {'read_memory':False, 'read_null':False, 'write_memory':False}}
    if interventions:
        flags.update(reads_disabled={'read_memory':False}, writes_disabled={'write_memory':False},
                     zero_values={'zero_memory_values':True}, forget_zero={'forget_override':0.},
                     forced_teacher={'memory_teacher':True})
    report = {}
    for name, policy in flags.items():
        summed, count, late, late_count, mass, mass_count = 0.,0,0.,0,0.,0
        for row in data:
            tokens = torch.tensor(row[None,:-1], device='cuda')
            target = torch.tensor(row[None,1:], device='cuda')
            output = model(tokens, diagnostics=True, **policy)
            loss = token_losses(model, output.hidden, target, COMMON['loss_chunk']).double()
            summed += loss.sum().item(); count += loss.numel()
            late += loss[model.capacity:].sum().item(); late_count += loss.numel()-model.capacity
            mass += output.diagnostics['memory_mass'].sum().item()
            mass_count += output.diagnostics['memory_mass'].numel()
        report[name] = {'ce':summed/count, 'tokens':count, 'post_eviction_ce':late/late_count,
                        'post_eviction_tokens':late_count, 'memory_mass':mass/mass_count}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('ordinary','full_block_teacher'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        parser.error('Slurm/CUDA required')
    identity = json.loads((args.root/'identity.json').read_text())
    for name, digest in identity['model_sha256'].items():
        if sha256(Path(identity['model_path'])/name) != digest:
            raise ValueError('pretrained asset changed')
    data = {}
    for split, entry in identity['splits'].items():
        path = args.root/'data'/entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError('corpus changed')
        data[split] = np.load(path, allow_pickle=False)
    torch.manual_seed(COMMON['seed']); torch.cuda.manual_seed_all(COMMON['seed'])
    model = RecurrentSinkLM(load_native(identity['model_path']), COMMON['capacity'], COMMON['timescales']).cuda()
    model.checkpointing = True
    opt = optimizer(model)
    sample = torch.tensor(data['train'][:1,:-1], device='cuda')
    calibration = model.calibrate([sample], target_mass=COMMON['initial_mass'])
    run = args.root / ('smoke' if args.smoke else 'runs/'+args.arm)
    run.mkdir(parents=True, exist_ok=False)
    def emit(event):
        print(json.dumps(event), flush=True)
        with (run/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
    atomic_json(run/'config.json', {'common':COMMON, 'arm':args.arm, 'calibration':calibration,
        'model':identity['model'], 'revision':identity['revision'], 'torch':str(torch.__version__),
        'slurm_job':os.environ['SLURM_JOB_ID'], 'new_parameters':sum(p.numel() for p in model.memory.parameters())})
    initial = evaluate(model, data['validation'][:1] if args.smoke else data['validation'])
    with torch.no_grad():
        ordinary_hidden = model(sample).hidden
        base_hidden = model(sample, read_memory=False, read_null=False).hidden
        drift = (ordinary_hidden-base_hidden).abs().max().item()
        if not torch.isfinite(ordinary_hidden).all() or drift > .05:
            raise AssertionError('initial hidden-state drift exceeds pilot tolerance')
    initial['max_initial_hidden_drift'] = drift
    atomic_json(run/'initial.json', initial)
    emit({'event':'startup','initial':initial,'calibration':calibration})
    started = time.monotonic()
    steps = 2 if args.smoke else COMMON['steps']
    backbone = {n:p.detach().cpu().clone() for n,p in model.base.named_parameters()}
    for step in range(1,steps+1):
        if (args.smoke and step==2) or step==COMMON['warmup_steps']+1:
            unchanged = all(torch.equal(p.detach().cpu(),backbone[n]) for n,p in model.base.named_parameters())
            if not unchanged:
                raise AssertionError('frozen warmup changed backbone')
            atomic_json(run/'transfer.json', {'backbone_unchanged':unchanged,'step':step-1})
            del backbone
            opt = optimizer(model, adapting=True, previous=opt)
        model.train(); opt.zero_grad(set_to_none=True)
        row = data['train'][step-1]
        tokens = torch.tensor(row[None,:-1], device='cuda')
        target = torch.tensor(row[None,1:], device='cuda')
        # Separate backward calls bound peak activation memory. Both arms
        # execute two passes and receive identical post-eviction target weights.
        output = model(tokens)
        ordinary = token_losses(model, output.hidden, target, COMMON['loss_chunk']).mean()
        ordinary.backward()
        ordinary_value = ordinary.item()
        del ordinary, output
        auxiliary = objective(model,tokens,target,teacher=args.arm=='full_block_teacher')
        auxiliary.backward()
        auxiliary_value = auxiliary.item()
        del auxiliary
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        emit({'event':'train','step':step,'ordinary_ce':ordinary_value,'auxiliary_ce':auxiliary_value,
              'gradient_norm':norm.item(),'unique_input_tokens':step*COMMON['length'],
              'elapsed_seconds':time.monotonic()-started,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    final = evaluate(model, data['validation'][:1] if args.smoke else data['validation'], interventions=not args.smoke)
    atomic_json(run/'validation.json',final)
    if not args.smoke:
        atomic_json(run/'test.json',evaluate(model,data['test'],interventions=True))
        torch.save({'model':model.state_dict(),'common':COMMON,'arm':args.arm,'identity':identity},run/'final.pt')
    atomic_json(run/'complete.json',{'status':'passed' if args.smoke else 'completed','steps':steps,
        'elapsed_seconds':time.monotonic()-started,'validation':final,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    emit({'event':'completed','steps':steps})


if __name__=='__main__':
    main()
