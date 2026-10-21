"""Frozen-checkpoint prefix-replacement screen; never trains or writes checkpoints."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from smollm_streaming_decoder import StreamingDecoder
from smollm_rebased_reference import RebasedReference

SEEDS = list(range(8, 14))
SOURCES = ['fineweb', 'cosmopedia', 'python', 'math', 'stackoverflow']
ROOTS = ['/cluster/users/alex116u1/data/smollm-full512-replication-20261009',
         '/cluster/users/alex116u1/data/smollm-warm70-20261014']
CAPACITY, PREDICTIONS = 512, 4096


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def training_root(seed):
    return Path(ROOTS[int(seed >= 11)])


def policies():
    return [(arm, disabled, prefix, rebase)
            for arm in ['standard', 'query_sink']
            for disabled in ([True] if arm == 'standard' else [False, True])
            for prefix in [0, 4] for rebase in [False, True]]


def label(policy):
    arm, disabled, prefix, rebase = policy
    return f'{arm}-{"off" if disabled else "on"}-prefix{prefix}-{"relative" if rebase else "absolute"}'


def prepare(out):
    identity = dict(checkpoints={}, data={}, numerical_source={})
    for seed in SEEDS:
        root = training_root(seed)
        pair = []
        for arm in ['standard', 'query_sink']:
            run = root / f'seeds/seed{seed}/runs/{arm}'
            config = json.loads((run / 'config.json').read_text())
            pair.append(config)
            identity['checkpoints'][f'{seed}-{arm}'] = dict(
                sha256=sha(run / 'final.pt'), config=config)
        ignored = {'variant', 'output_dir'}
        mismatches = [k for k in set(pair[0]) | set(pair[1])
                      if k not in ignored and pair[0].get(k) != pair[1].get(k)]
        assert not mismatches, mismatches
        assert pair[0]['shared_initial_weights_sha256'] == pair[1]['shared_initial_weights_sha256']
    for root_name in ROOTS:
        root = Path(root_name)
        manifest = json.loads((root / 'data/manifest.json').read_text())
        identity['numerical_source'][root_name] = {
            name: sha(root / 'release/src/virtualsink/uptraining' / name)
            for name in ['native.py', 'qwen.py']}
        for source in SOURCES:
            entry = manifest['sources'][source]['splits']['test']
            path = root / 'data' / entry['file']
            assert sha(path) == entry['sha256']
            ids = np.memmap(path, dtype='<u4', mode='r')[:PREDICTIONS + 1]
            assert len(ids) == PREDICTIONS + 1
            record = dict(file=entry['file'], sha256=entry['sha256'],
                          selected_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
            if source in identity['data']:
                assert identity['data'][source] == record
            identity['data'][source] = record
    assert len({json.dumps(v, sort_keys=True) for v in identity['numerical_source'].values()}) == 1
    write(out / 'identity.json', identity)
    print('IDENTITY_PINNED', flush=True)


def load(seed, arm, out, device):
    root = training_root(seed)
    sys.path.insert(0, str(root / 'release/src'))
    from virtualsink.pretraining.recipe import initialize, MODEL_CONFIG
    identity = json.loads((out / 'identity.json').read_text())
    for name, expected in identity['numerical_source'][str(root)].items():
        assert sha(root / 'release/src/virtualsink/uptraining' / name) == expected
    path = root / f'seeds/seed{seed}/runs/{arm}/final.pt'
    pinned = identity['checkpoints'][f'{seed}-{arm}']
    assert sha(path) == pinned['sha256']
    payload = torch.load(path, map_location='cpu')
    assert payload['config'] == pinned['config']
    assert payload['progress']['step'] == pinned['config']['steps']
    model = initialize(seed, 32, dict(MODEL_CONFIG, max_position_embeddings=512),
                       sink_weight_std=.02, sink_bias=4.)
    model.load_state_dict(payload['model'])
    del payload
    return model.to(device).eval()


def set_disabled(model, disabled):
    for layer in model.model.layers:
        layer.self_attn.sink_disabled = disabled


def tokens_for(seed, source, out):
    identity = json.loads((out / 'identity.json').read_text())['data'][source]
    path = training_root(seed) / 'data' / identity['file']
    assert sha(path) == identity['sha256']
    ids = np.memmap(path, dtype='<u4', mode='r')[:PREDICTIONS + 1]
    assert hashlib.sha256(ids.tobytes()).hexdigest() == identity['selected_sha256']
    return torch.tensor(ids.astype(np.int64), device='cuda')


@torch.no_grad()
def validate_model(model, tokens, capacity=512, length=641):
    """Real-checkpoint causal/window parity and independent rebasing oracle."""
    from virtualsink.uptraining.native import set_window
    checks = {}
    for disabled in [False, True]:
        set_disabled(model, disabled)
        set_window(model, capacity)
        expected = model.lm_head(model.model(tokens[:length][None]).last_hidden_state)[0]
        decoder = StreamingDecoder(model, capacity)
        maximum = 0.
        for i, token in enumerate(tokens[:length]):
            actual, *_ = decoder.step(token)
            maximum = max(maximum, (actual - expected[i]).abs().max().item())
            torch.testing.assert_close(actual, expected[i], atol=.003, rtol=.0001)
            assert all(k.shape[2] <= capacity for k, _ in decoder.cache)
        checks[f'parallel-disabled-{disabled}'] = maximum
        del expected, decoder
    set_window(model, None)
    set_disabled(model, True)
    decoder, oracle = StreamingDecoder(model, capacity, 4, True), RebasedReference(model, capacity, 4)
    maximum = 0.
    for token in tokens[:length]:
        actual, *_ = decoder.step(token)
        expected = oracle.step(token)
        maximum = max(maximum, (actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=.003, rtol=.0001)
    checks['delta_rotation_oracle'] = maximum
    return checks


@torch.no_grad()
def evaluate(out, task):
    seed, source = SEEDS[task // 5], SOURCES[task % 5]
    tokens = tokens_for(seed, source, out)
    unit = out / 'results' / f'seed{seed}-{source}'
    unit.mkdir(exist_ok=False)
    identity_sha = sha(out / 'identity.json')
    for arm in ['standard', 'query_sink']:
        model = load(seed, arm, out, 'cuda')
        for policy in [p for p in policies() if p[0] == arm]:
            _, disabled, prefix, rebase = policy
            set_disabled(model, disabled)
            decoder = StreamingDecoder(model, CAPACITY, prefix, rebase)
            losses, masses = [], []
            started = time.monotonic()
            for position, token in enumerate(tokens[:-1]):
                logits, mass, *_ = decoder.step(token)
                loss = torch.nn.functional.cross_entropy(logits[None], tokens[position + 1:position + 2])
                losses.append(loss.item())
                masses.append(mass.mean().item())
                assert all(k.shape[2] == min(position + 1, CAPACITY) for k, _ in decoder.cache)
                if (position + 1) % 256 == 0:
                    elapsed = time.monotonic() - started
                    status = dict(seed=seed, source=source, policy=label(policy),
                                  predictions=position+1, total=PREDICTIONS,
                                  seconds=elapsed, tokens_per_second=(position+1)/elapsed)
                    write(unit / 'progress.json', status)
                    print(json.dumps(status), flush=True)
            assert np.isfinite(losses).all()
            write(unit / (label(policy) + '.json'), dict(
                seed=seed, source=source, policy=policy, identity_sha256=identity_sha,
                per_token_ce=losses, per_token_sink_mass=masses,
                elapsed_seconds=time.monotonic()-started))
            del decoder
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write(unit / 'complete.json', dict(seed=seed, source=source, identity_sha256=identity_sha))


def summarize(out):
    rows, missing = [], []
    identity_sha = sha(out / 'identity.json')
    for seed in SEEDS:
        sources = {}
        for source in SOURCES:
            unit = out / 'results' / f'seed{seed}-{source}'
            if not (unit / 'complete.json').exists():
                missing.append(str(unit))
                continue
            scores = {}
            for policy in policies():
                path = unit / (label(policy) + '.json')
                value = json.loads(path.read_text())
                assert value['identity_sha256'] == identity_sha
                assert value['policy'] == list(policy) and value['seed'] == seed and value['source'] == source
                ce = np.asarray(value['per_token_ce'])
                assert len(ce) == PREDICTIONS and np.isfinite(ce).all()
                scores[label(policy)] = dict(pre_eviction=float(ce[:512].mean()),
                    post_eviction=float(ce[512:].mean()), late=float(ce[2048:].mean()),
                    post_eviction_ppl=math.exp(float(ce[512:].mean())))
            sources[source] = scores
        if len(sources) != 5:
            continue
        mean = {key: float(np.mean([scores[key]['post_eviction'] for scores in sources.values()]))
                for key in sources[SOURCES[0]]}
        contrasts = {}
        for rebase in [False, True]:
            get = lambda arm, off, p: mean[label((arm, off, p, rebase))]
            contrasts['relative' if rebase else 'absolute'] = dict(
                standard_prefix_rescue=get('standard', True, 0)-get('standard', True, 4),
                virtual_prefix_rescue_enabled=get('query_sink', False, 0)-get('query_sink', False, 4),
                virtual_prefix_rescue_disabled=get('query_sink', True, 0)-get('query_sink', True, 4),
                virtual_fifo_enabling_benefit=get('query_sink', True, 0)-get('query_sink', False, 0),
                virtual_fifo_minus_enabled_prefix=get('query_sink', False, 0)-get('query_sink', False, 4))
        rows.append(dict(seed=seed, mean_ce=mean, contrasts=contrasts, sources=sources))
    result = dict(complete=not missing, missing=missing, seeds=rows,
                  note='Equal weight across five fixed source streams; descriptive screen. Shared dataset across seeds. No independent document or seed confidence claim. Prefix rescue absent in the standard model means the replacement test is uninformative for that seed. Absolute/relative policies are separate; relative virtual queries also change sink scores.')
    write(out / ('aggregate.json' if not missing else 'INCOMPLETE.json'), result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'validate', 'evaluate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    if args.mode == 'prepare':
        prepare(args.root)
    elif args.mode == 'summarize':
        summarize(args.root)
    else:
        assert torch.cuda.is_available()
        task = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if args.mode == 'evaluate':
            evaluate(args.root, task)
        else:
            seed = SEEDS[task]
            tokens = tokens_for(seed, 'fineweb', args.root)
            checks = {}
            for arm in ['standard', 'query_sink']:
                model = load(seed, arm, args.root, 'cuda')
                checks[arm] = validate_model(model, tokens)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print('PARITY_PASSED', seed, arm, checks[arm], flush=True)
            write(args.root / 'validation' / f'seed{seed}.json', checks)


if __name__ == '__main__':
    main()
