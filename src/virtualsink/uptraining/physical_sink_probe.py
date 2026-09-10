"""Frozen physical-token attention measurements; no checkpoint writes."""
import json
import os
from pathlib import Path

import torch

from . import native
from .data import TokenBlocks, sha256
from .qwen import disable_sinks


@torch.no_grad()
def probe(model, data, blocks, output, label):
    original = native.sink_attention
    totals, dominant, calls = [], [], 0
    layers = len(model.model.layers)

    def measured(q, k, v, w, b, chunk_size=128, recompute=True, window_size=None):
        nonlocal calls
        assert window_size is None and q.shape[0] == 1 and q.shape[2] == 256
        scores = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
        positions = torch.arange(q.shape[2], device=q.device)
        scores.masked_fill_(positions[None, :] > positions[:, None], -torch.inf)
        sink = (q.float() * w.float()[None, :, None]).sum(-1) + b.float()[None, :, None]
        weights = torch.softmax(torch.cat((scores, sink[..., None]), -1), -1)
        # Late queries prevent the trivial early causal-position advantage.
        late = weights[0, :, 128:256, :]
        layer = calls % layers
        means = late.mean(1).cpu()
        frequency = (late[:, :, :4].sum(-1) > .5).float().mean(1).cpu()
        if len(totals) < layers:
            totals.append(means)
            dominant.append(frequency)
        else:
            totals[layer] += means
            dominant[layer] += frequency
        result = weights[..., :-1] @ v.float()
        gates = 1 - weights[..., -1]
        if calls == 0:
            ref, ref_gates = original(q, k, v, w, b, chunk_size, False, None)
            torch.testing.assert_close(result, ref, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(gates, ref_gates, rtol=2e-5, atol=2e-5)
        calls += 1
        return result.to(v.dtype), gates

    native.sink_attention = measured
    try:
        for index in range(blocks):
            model.model(data[index][None].cuda())
            if (index + 1) % 16 == 0:
                print(f'{label}: {index+1}/{blocks}', flush=True)
    finally:
        native.sink_attention = original
    assert calls == blocks * layers
    mass = torch.stack(totals) / blocks
    frequency = torch.stack(dominant) / blocks
    first = mass[:, :, 0]
    first4 = mass[:, :, :4].sum(-1)
    result = {
        'blocks': blocks, 'query_positions_inclusive': [128, 255],
        'mean_mass_by_layer_head_key_then_virtual': mass.tolist(),
        'first4_majority_query_fraction_by_layer_head': frequency.tolist(),
        'mean_first_token_mass': first.mean().item(),
        'mean_first4_mass': first4.mean().item(),
        'mean_virtual_mass': mass[:, :, -1].mean().item(),
        'heads': first.numel(),
        'heads_first_token_mean_over_10pct': (first > .1).sum().item(),
        'heads_first4_mean_over_50pct': (first4 > .5).sum().item(),
        'top_first_token_heads': [
            {'layer_zero_based': i // first.shape[1], 'head_zero_based': i % first.shape[1],
             'first_token_mass': first.flatten()[i].item(),
             'first4_mass': first4.flatten()[i].item(),
             'first4_majority_query_fraction': frequency.flatten()[i].item()}
            for i in first.flatten().argsort(descending=True)[:12].tolist()],
    }
    (output / f'{label}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}), flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run in a Slurm GPU allocation')
    base = Path.home() / 'data/phase3-k80-20260909'
    path = base / 'runs/qwen05/last.pt'
    output = Path.home() / 'data/qwen-final-physical-sinks-20260910'
    output.mkdir(exist_ok=False)
    saved = torch.load(path, map_location='cpu')
    config = saved['config']
    assert saved['step'] == 1000 and config['seq_len'] == 256
    assert not config.get('window_size') and config['variant'] == 'query_sink'
    manifest = config['manifest']
    test = Path(config['data_dir']) / manifest['splits']['test']['file']
    assert sha256(test) == manifest['splits']['test']['sha256']
    model = native.load_native(manifest['model_path'], 64).eval().cuda()
    data = TokenBlocks(test, 256, model.config.vocab_size)
    blocks = min(128, len(data))
    metadata = {'checkpoint': str(path), 'checkpoint_step': saved['step'],
                'checkpoint_sha256': sha256(path), 'training_config': config,
                'blocks': blocks, 'job_id': os.environ['SLURM_JOB_ID'],
                'interpretation': 'Physical attention concentration, not proof of no-op function. '
                'Position zero is the first packed token, not necessarily BOS. '
                'Mass uses the joint physical-plus-virtual denominator; no physical renormalization.'}
    (output / 'config.json').write_text(json.dumps(metadata, indent=2))
    disable_sinks(model)
    probe(model, data, blocks, output, 'original_standard')
    model.load_state_dict(saved['model'])
    del saved
    disable_sinks(model, False)
    probe(model, data, blocks, output, 'final_virtual_enabled')


if __name__ == '__main__':
    main()
