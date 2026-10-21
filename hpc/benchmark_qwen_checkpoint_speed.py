"""Paired training benchmark from one immutable ordinary-memory checkpoint."""
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.qwen05_long import COMMON, make_optimizer, set_rates, train_pair
from virtualsink.uptraining.recurrent_data import TokenDocuments
from virtualsink.uptraining.recurrent_sink import RecurrentSinkLM
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM
from virtualsink.uptraining.train_recurrent import load_checkpoint, pack_state, unpack_state


def compare_states(actual, expected):
    maximum = 0.
    for sa, sb in zip(actual, expected):
        assert (sa is None) == (sb is None)
        if sa is None:
            continue
        assert (sa['position'], sa['capacity']) == (sb['position'], sb['capacity'])
        for la, lb in zip(sa['layers'], sb['layers']):
            for name in ('key', 'value', 'cell', 'exposed'):
                a, b = la[name], lb[name]
                torch.testing.assert_close(a, b, atol=3e-4, rtol=1e-3, msg=name)
                maximum = max(maximum, float((a-b).abs().max()))
    return maximum


def run_arm(arm, saved, model_config, data, output, baseline=None):
    cls = RecurrentSinkLM if arm == 'current' else OptimizedRecurrentSinkLM
    model = cls(NativeLM(model_config, 32), COMMON['capacity'], COMMON['timescales'])
    model.load_state_dict(saved['model'])
    model = model.cuda()
    model.checkpointing = True
    optimizer = make_optimizer(model, 'ordinary', saved['adapting'])
    optimizer.load_state_dict(saved['optimizer'])
    states = tuple(unpack_state(s, 'cuda') for s in saved['states'])
    cursor = tuple(saved['cursor'])
    torch.set_rng_state(saved['rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    torch.cuda.reset_peak_memory_stats()
    samples, audits = [], []
    gradient_reference = None
    gradient_error = None
    for index in range(6):
        step = saved['step']+index
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        set_rates(optimizer, step, saved['adapting'])
        losses, count, states, cursor = train_pair(model, data, cursor, states, 'ordinary', 'cuda')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        # Audit copies and comparisons occur outside the timed update.
        packed = [pack_state(s) for s in states]
        state_error = 0.
        if baseline is not None:
            state_error = compare_states(packed, baseline['states'][index])
            expected = baseline['samples'][index]
            assert list(cursor) == expected['cursor'] and count == expected['tokens']
            torch.testing.assert_close(torch.tensor(losses), torch.tensor(expected['losses']), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(norm.cpu(), torch.tensor(expected['gradient_norm']), atol=1e-3, rtol=1e-3)
        if index == 0:
            if baseline is None:
                gradient_reference = {name: p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
            else:
                difference_sq, reference_sq, maximum = 0., 0., 0.
                actual_names = set()
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    actual_names.add(name)
                    ref = baseline['gradients'][name]
                    delta = p.grad.detach().cpu()-ref
                    maximum = max(maximum, float(delta.abs().max()))
                    difference_sq += float(delta.double().square().sum())
                    reference_sq += float(ref.double().square().sum())
                assert actual_names == set(baseline['gradients'])
                relative = math.sqrt(difference_sq/max(reference_sq, 1e-30))
                if maximum > 1e-5 or relative > 1e-3:
                    raise AssertionError('parameter gradient parity failed: '+str((maximum,relative)))
                gradient_error = dict(max_absolute=maximum, relative_l2=relative, clipped=True)
        audits.append(packed)
        row = dict(arm=arm, step=step+1, measured=index>0, seconds=seconds, tokens=count,
                   losses=losses, gradient_norm=float(norm), cursor=list(cursor),
                   context_position=None if states[0] is None else states[0].position,
                   max_state_difference=state_error)
        samples.append(row)
        print(json.dumps(row), flush=True)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    measured = samples[1:]
    result = dict(arm=arm, checkpoint_step=saved['step'], checkpoint_cursor=saved['cursor'],
        config=COMMON, samples=samples, mean_seconds=statistics.mean(r['seconds'] for r in measured),
        median_seconds=statistics.median(r['seconds'] for r in measured),
        tokens_per_second=sum(r['tokens'] for r in measured)/sum(r['seconds'] for r in measured),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gradient_parity=gradient_error)
    (output/(arm+'.json')).write_text(json.dumps(result,indent=2))
    print('ARM_COMPLETE '+json.dumps({k:v for k,v in result.items() if k not in ('samples','config')}),flush=True)
    return result, dict(states=audits, gradients=gradient_reference, samples=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires Slurm GPU')
    torch.set_num_threads(4)
    saved = load_checkpoint(args.root/'snapshot.pt')
    assert saved['identity']['arm'] == 'ordinary' and saved['adapting']
    assert saved['identity']['common'] == COMMON
    manifest = json.loads((args.training_root/'data/manifest.json').read_text())
    config = json.loads((Path(manifest['model_path'])/'config.json').read_text())
    data = TokenDocuments(args.training_root/'data', 'train', config['vocab_size'])
    info = dict(job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, checkpoint_step=saved['step'],
        checkpoint_input_tokens=saved['input_tokens'], checkpoint_cursor=saved['cursor'],
        checkpoint_mtime=(args.root/'snapshot.pt').stat().st_mtime,
        scope='one warmup + five measured full training updates per arm; no production writes')
    (args.root/'checkpoint-info.json').write_text(json.dumps(info,indent=2))
    print('CHECKPOINT_LOADED '+json.dumps(info),flush=True)
    current, baseline = run_arm('current', saved, config, data, args.root)
    gc.collect()
    torch.cuda.empty_cache()
    optimized, _ = run_arm('optimized', saved, config, data, args.root, baseline)
    cs, osamples = current['samples'][1:], optimized['samples'][1:]
    full = [(a,b) for a,b in zip(cs,osamples) if a['tokens']==512]
    result = dict(**info, mean_speedup=current['mean_seconds']/optimized['mean_seconds'],
        median_speedup=current['median_seconds']/optimized['median_seconds'],
        time_reduction_percent=100*(1-optimized['mean_seconds']/current['mean_seconds']),
        current_seconds=current['mean_seconds'], optimized_seconds=optimized['mean_seconds'],
        current_tokens_per_second=current['tokens_per_second'], optimized_tokens_per_second=optimized['tokens_per_second'],
        current_peak_allocated_gb=current['peak_allocated_bytes']/1e9, optimized_peak_allocated_gb=optimized['peak_allocated_bytes']/1e9,
        full_length_update_count=len(full), full_length_speedup=sum(a['seconds'] for a,b in full)/sum(b['seconds'] for a,b in full) if full else None,
        parity='PASS', gradient_parity=optimized['gradient_parity'],
        max_state_difference=max(r['max_state_difference'] for r in optimized['samples']),
        max_loss_difference=max(abs(a-b) for c,o in zip(current['samples'],optimized['samples']) for a,b in zip(c['losses'],o['losses'])))
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2))
    print('BENCHMARK_COMPLETE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
