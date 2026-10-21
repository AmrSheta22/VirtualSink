"""Read-only factorial test on Pythia checkpoints with a known streaming gap."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

REFERENCE = Path('/cluster/users/alex116u1/data/pythia410-trained-streaming-20261014-r2')
ASSETS = Path('/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3')
ARMS = ['pretrained', 'standard', 'legacy', 'calibrated']
BUDGET = 65536
CAPACITY = 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def policies():
    return [(arm, disabled, prefix, relative) for arm in ARMS
            for disabled in ([True] if arm in ['pretrained', 'standard'] else [False, True])
            for prefix in [0, 4] for relative in [False, True]]


def label(policy):
    arm, disabled, prefix, relative = policy
    return f'{arm}-{"off" if disabled else "on"}-prefix{prefix}-{"relative" if relative else "absolute"}'


def prepare(root):
    manifest = json.loads((ASSETS/'manifest.json').read_text())
    model = Path(manifest['models']['pythia410']['path'])
    identity = dict(model=manifest['models']['pythia410'], checkpoints={}, assets={}, references={})
    paths = [model/name for name in ['config.json','model.safetensors','tokenizer.json']]
    paths += [REFERENCE/'data/pythia28-test.npz']
    for path in paths:
        identity['assets'][str(path)] = sha(path)
    for arm in ARMS[1:]:
        checkpoint = REFERENCE/'pilot'/arm/'final-updates.pt'
        pinned = json.loads((REFERENCE/'results'/f'{arm}-fifo/identity.json').read_text())
        assert sha(checkpoint) == pinned['checkpoint_sha256']
        assert pinned['checkpoint_step'] == 256
        assert pinned['test_stream_sha256'] == identity['assets'][str(REFERENCE/'data/pythia28-test.npz')]
        identity['checkpoints'][arm] = dict(path=str(checkpoint), sha256=pinned['checkpoint_sha256'])
        means = {}
        for policy in ['fifo','streamingllm']:
            path = REFERENCE/'results'/f'{arm}-{policy}'/'losses.npy'
            losses = np.load(path)
            assert len(losses) == 400000 and np.isfinite(losses).all()
            means[policy] = float(losses[1024:BUDGET].mean())
            identity['references'][f'{arm}-{policy}'] = dict(path=str(path), sha256=sha(path), ce=means[policy])
        assert means['fifo'] - means['streamingllm'] > .2, (arm, means)
    write(root/'identity.json', identity)
    print('POSITIVE_CONTROL_AND_IDENTITIES_PINNED', json.dumps(identity['references']), flush=True)


def load(root, arm):
    from transformers import AutoModelForCausalLM
    from positive_pythia_adapt import adapt
    identity = json.loads((root/'identity.json').read_text())
    for path, expected in identity['assets'].items():
        assert sha(path) == expected, path
    torch.manual_seed(1729)
    model = AutoModelForCausalLM.from_pretrained(identity['model']['path'],
        torch_dtype=torch.float32, use_safetensors=True, local_files_only=True)
    adapt(model, 'standard' if arm == 'pretrained' else arm)
    if arm != 'pretrained':
        spec = identity['checkpoints'][arm]
        assert sha(spec['path']) == spec['sha256']
        updates = torch.load(spec['path'], map_location='cpu')
        assert set(updates) == {n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(updates, strict=False)
        del updates
    return model.cuda().eval()


def disabled(model, flag):
    for layer in model.gpt_neox.layers:
        layer.attention.sink_disabled = flag


def first_tokens(length=1153):
    with np.load(REFERENCE/'data/pythia28-test.npz') as data:
        return torch.tensor(data['book000'][:length], device='cuda')


@torch.no_grad()
def validate(root, arm):
    from positive_neox_decoder import NeoXDecoder
    from streaming_llm.kv_cache import StartRecentKVCache
    from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention
    rows = {}
    tokens = first_tokens()
    # Fresh model per policy avoids mutation of absolute attention by the official shifter.
    for policy in [p for p in policies() if p[0] == arm]:
        _, off, prefix, relative = policy
        model = load(root, arm)
        disabled(model, off)
        if relative:
            enable_gpt_neox_pos_shift_attention(model)
        decoder = NeoXDecoder(model, CAPACITY, prefix, relative)
        cache = StartRecentKVCache(prefix, CAPACITY-prefix)
        past, maximum = None, 0.
        for position, token in enumerate(tokens):
            index = (0 if past is None else past[0][0].shape[2]) if relative else position
            output = model(token.reshape(1,1), past_key_values=past,
                position_ids=torch.tensor([[index]], device='cuda'), use_cache=True)
            past = cache(output.past_key_values)
            actual = decoder.step(token)
            maximum = max(maximum, (actual-output.logits[0,0]).abs().max().item())
            torch.testing.assert_close(actual, output.logits[0,0], atol=.003, rtol=.0001)
            assert all(k.shape[2] <= CAPACITY for k,v in decoder.cache)
        rows[label(policy)] = dict(tokens=len(tokens), maximum_absolute_logit_error=maximum)
        print('PARITY_PASSED', label(policy), maximum, flush=True)
        del model, decoder, past, output
        torch.cuda.empty_cache()
    write(root/'validation'/f'{arm}.json', rows)


@torch.no_grad()
def evaluate(root, task):
    from positive_neox_decoder import NeoXDecoder
    policy = policies()[task]
    arm, off, prefix, relative = policy
    model = load(root, arm)
    disabled(model, off)
    decoder = NeoXDecoder(model, CAPACITY, prefix, relative)
    folder = root/'results'/label(policy)
    folder.mkdir(exist_ok=False)
    losses, boundaries, started = [], [], time.monotonic()
    with np.load(REFERENCE/'data/pythia28-test.npz') as data:
        for book in sorted(data.files):
            tokens = torch.tensor(data[book], device='cuda')
            boundaries.append(dict(book=book, prediction_start=len(losses)))
            for index in range(len(tokens)-1):
                logits = decoder.step(tokens[index])
                ce = torch.nn.functional.cross_entropy(logits[None], tokens[index+1:index+2]).item()
                assert math.isfinite(ce)
                losses.append(ce)
                if len(losses)%512 == 0:
                    status = dict(policy=label(policy), predictions=len(losses), total=BUDGET,
                        seconds_per_prediction=(time.monotonic()-started)/len(losses))
                    write(folder/'progress.json', status)
                    print(json.dumps(status), flush=True)
                if len(losses) == BUDGET:
                    break
            if len(losses) == BUDGET:
                break
    assert len(losses) == BUDGET
    losses = np.asarray(losses, dtype='<f8')
    # Reproduce the existing positive control and enabled sink baselines exactly.
    if arm != 'pretrained' and not (arm != 'standard' and off):
        old_policy = 'fifo' if prefix == 0 and not relative else 'streamingllm' if prefix == 4 and relative else None
        if old_policy:
            original = np.load(REFERENCE/'results'/f'{arm}-{old_policy}'/'losses.npy')[:BUDGET]
            assert np.max(np.abs(original-losses)) < .003
    np.save(folder/'losses.npy', losses)
    write(folder/'complete.json', dict(policy=list(policy), identity_sha256=sha(root/'identity.json'),
        predictions=BUDGET, boundaries=boundaries, elapsed_seconds=time.monotonic()-started))


def contrast(means, relative):
    get = lambda arm, off, prefix: means[label((arm,off,prefix,relative))]
    result = dict(pretrained_prefix_rescue=get('pretrained',True,0)-get('pretrained',True,4),
        standard_prefix_rescue=get('standard',True,0)-get('standard',True,4))
    for arm in ['legacy','calibrated']:
        result[arm] = dict(prefix_rescue_disabled=get(arm,True,0)-get(arm,True,4),
            prefix_rescue_enabled=get(arm,False,0)-get(arm,False,4),
            fifo_sink_enabling_benefit=get(arm,True,0)-get(arm,False,0),
            enabled_fifo_minus_standard_prefix=get(arm,False,0)-get('standard',True,4))
    return result


def summarize(root):
    rows, missing, boundaries = {}, [], None
    for policy in policies():
        folder = root/'results'/label(policy)
        if not (folder/'complete.json').exists():
            missing.append(label(policy))
            continue
        complete = json.loads((folder/'complete.json').read_text())
        assert complete['policy'] == list(policy) and complete['identity_sha256'] == sha(root/'identity.json')
        if boundaries is None:
            boundaries = complete['boundaries']
        assert boundaries == complete['boundaries']
        values = np.load(folder/'losses.npy')
        assert values.shape == (BUDGET,) and np.isfinite(values).all()
        rows[label(policy)] = {name:dict(ce=float(part.mean()),ppl=math.exp(float(part.mean())))
            for name,part in [('first1024',values[:1024]),('post_eviction',values[1024:]),('late',values[10240:])]}
    result = dict(complete=not missing, missing=missing, policies=rows, boundaries=boundaries)
    if not missing:
        means = {k:v['post_eviction']['ce'] for k,v in rows.items()}
        result['contrasts'] = {name:contrast(means, relative) for name,relative in [('absolute',False),('relative',True)]}
    write(root/('INCOMPLETE.json' if missing else 'aggregate.json'), result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','validate','evaluate','summarize'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if args.mode == 'prepare': prepare(args.root)
    elif args.mode == 'summarize': summarize(args.root)
    elif args.mode == 'validate': validate(args.root, ARMS[int(os.environ['SLURM_ARRAY_TASK_ID'])])
    else: evaluate(args.root, int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__ == '__main__':
    main()
