"""Slurm-only Qwen block-memory numerical gate and small frozen-backbone pilot."""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from virtualsink.uptraining.block_memory import BlockMemoryLM
from virtualsink.uptraining.data import sha256
from virtualsink.uptraining.native import load_native
from virtualsink.uptraining.recurrent_data import TokenDocuments
from virtualsink.uptraining.train_recurrent import token_losses


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs)), flush=True)


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def check_source(root):
    manifest = json.loads((root / 'source-sha256.json').read_text())
    for name, expected in manifest.items():
        if sha256(root / 'release' / name) != expected:
            raise ValueError('frozen source changed: ' + name)


def check_pretrained(manifest):
    for name, expected in manifest['pretrained_sha256'].items():
        if sha256(Path(manifest['model_path']) / name) != expected:
            raise ValueError('pretrained file changed: ' + name)
    if (manifest['model'], manifest['revision']) != (
            'Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987'):
        raise ValueError('unexpected pretrained backbone')


@torch.no_grad()
def evaluate(model, data, indices, condition, flags, output):
    model.eval()
    totals = {}
    for document in indices:
        state = None
        limit = data.length(document) - 1  # full article; no hidden 4K cap
        for start in range(0, limit, 256):
            inputs, targets = data.segment(document, start, min(256, limit - start))
            result = model(inputs.cuda(), state, diagnostics=True, **flags)
            state = result.state
            losses = token_losses(model, result.hidden, targets.cuda(), 8).double()
            positions = torch.arange(start, start + inputs.shape[1], device=losses.device)
            for name, lower, upper in [('all', 0, limit), ('post_publication', 255, limit),
                ('0-4K', 0, 4096), ('4-8K', 4096, 8192),
                ('8-12K', 8192, 12288), ('12-16K', 12288, 16384)]:
                selected = losses[(positions >= lower) & (positions < upper)]
                record = totals.setdefault(name, [0., 0])
                record[0] += selected.sum().item()
                record[1] += selected.numel()
            row = dict(condition=condition, document=int(document), start=start,
                tokens=losses.numel(), losses=losses.cpu().tolist(),
                mean_role_mass=result.diagnostics.mean((0, 1, 2, 3)).cpu().tolist(),
                slots=state.layers[0].memory_key.shape[2])
            output.write(json.dumps(row) + '\n')
            output.flush()
        emit('article_scored', condition=condition, document=int(document), tokens=limit)
    return {name: dict(tokens=n, ce=loss / n if n else None) for name, (loss, n) in totals.items()}


def gate(root, data_dir, manifest):
    torch.manual_seed(1729)
    model = BlockMemoryLM(load_native(manifest['model_path']), capacity=16).cuda()
    model.parameter_groups('memory')
    data = TokenDocuments(data_dir, 'train', model.base.config.vocab_size)
    inputs, targets = data.segment(0, 0, 65)
    inputs, targets = inputs.cuda(), targets.cuda()
    # Appreciable reads exercise memory arithmetic, not only near-disabled parity.
    with torch.no_grad():
        for bank in model.memory:
            bank.role_bias.fill_(-2.)
            bank.null_bias.fill_(-4.)
    model.eval()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        full = model(inputs, diagnostics=True)
        expected = model.base.lm_head(full.hidden)
        state, outputs = None, []
        for token in inputs.unbind(1):
            logits, state, _ = model.step(token, state)
            outputs.append(logits)
        actual = torch.stack(outputs, 1)
        parity_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=.003, rtol=1e-4)
        for a, b in zip(full.state.layers, state.layers):
            for left, right in zip(a.tensors(), b.tensors()):
                torch.testing.assert_close(left, right, atol=.003, rtol=1e-4)
            assert a.memory_key.shape[2] == 9
        masses = full.diagnostics.mean((0, 1, 2, 3)).cpu().tolist()
        assert sum(masses[1:]) > 0
    emit('pretrained_parity_passed', max_logit_error=parity_error, mean_role_mass=masses)
    del full, expected, state, actual, outputs, logits
    model.train()
    model.checkpointing = True
    optimizer = torch.optim.AdamW(model.parameter_groups('memory'), foreach=False)
    rows = []
    for step in range(2):
        tick = time.monotonic()
        before = [bank.pool_query.detach().clone() for bank in model.memory]
        optimizer.zero_grad(set_to_none=True)
        first = model(inputs[:, :35])
        second = model(inputs[:, 35:], first.state)
        loss = torch.cat((token_losses(model, first.hidden, targets[:, :35], 8),
                          token_losses(model, second.hidden, targets[:, 35:], 8))).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite gate loss')
        loss.backward()
        for name, parameter in model.memory.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0:
                raise AssertionError('missing/nonfinite/zero gradient: ' + name)
        torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        updates = [(bank.pool_query.detach() - old).abs().sum().item() for bank, old in zip(model.memory, before)]
        assert all(value > 0 for value in updates)
        torch.cuda.synchronize()
        row = dict(step=step + 1, ce=loss.item(), seconds=time.monotonic() - tick,
                   pool_query_update_l1=updates)
        rows.append(row)
        emit('gate_optimizer_step', **row)
        del first, second, loss, before
    report = dict(status='passed', job=os.environ['SLURM_JOB_ID'], model=manifest['model'],
        revision=manifest['revision'], gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        capacity=16, tokens=65, max_logit_error=parity_error, mean_role_mass=masses,
        steps=rows, elapsed_seconds=time.monotonic() - started,
        new_parameters=sum(p.numel() for p in model.memory.parameters()),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), scope='correctness and profile only')
    save_json(root / 'gate.json', report)
    emit('BLOCK_MEMORY_GPU_GATE_PASSED')


def pilot(root, data_dir, manifest, arm, steps):
    if json.loads((root / 'gate.json').read_text())['status'] != 'passed':
        raise RuntimeError('GPU gate not passed')
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    model = BlockMemoryLM(load_native(manifest['model_path']), capacity=128).cuda()
    groups = model.parameter_groups('memory', memory_lr=.01)
    if arm == 'null':
        for bank in model.memory:
            bank.pool_query.requires_grad_(False)
            bank.role_bias.requires_grad_(False)
        groups[0]['params'] = [p for p in groups[0]['params'] if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    model.checkpointing = True
    train = TokenDocuments(data_dir, 'train', model.base.config.vocab_size)
    flags = {} if arm == 'block' else dict(read_memory=False)
    run = root / arm
    run.mkdir(exist_ok=False)
    cursor, state, tokens_seen = (0, 0), None, 0
    started = time.monotonic()
    with (run / 'train.jsonl').open('w') as output:
        for step in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            objective, count = None, 0
            # 512-token connected graph across two segments; forward state survives
            # optimizer boundaries and resets at authoritative article boundaries.
            for _ in range(2):
                document, offset = cursor
                if document >= len(train):
                    raise RuntimeError('training corpus exhausted; do not repeat')
                inputs, targets = train.segment(document, offset, 256)
                result = model(inputs.cuda(), state, **flags)
                losses = token_losses(model, result.hidden, targets.cuda(), 8)
                objective = losses.sum() if objective is None else objective + losses.sum()
                count += losses.numel()
                offset += inputs.shape[1]
                state = result.state
                if offset == train.length(document) - 1:
                    document, offset, state = document + 1, 0, None
                cursor = document, offset
            loss = objective / count
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite pilot objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.memory.parameters() if p.requires_grad],
                                                 1., error_if_nonfinite=True)
            optimizer.step()
            state = None if state is None else state.detach()
            tokens_seen += count
            row = dict(step=step + 1, ce=loss.item(), tokens=tokens_seen, cursor=cursor,
                       gradient_norm=norm.item(), seconds=time.monotonic() - tick)
            output.write(json.dumps(row) + '\n'); output.flush()
            emit('pilot_step', arm=arm, **row)
            if (step + 1) % 16 == 0 or step + 1 == steps:
                temporary = run / 'checkpoint.tmp'
                torch.save(dict(memory=model.memory.state_dict(), architecture=model.get_extra_state(),
                    optimizer=optimizer.state_dict(), step=step + 1, cursor=cursor, state=state,
                    input_tokens=tokens_seen, seed=1729, source=manifest), temporary)
                temporary.replace(run / 'last.pt')
            del result, losses, loss, objective
    reports = {}
    with (run / 'scores.jsonl').open('w') as output:
        for split in ('validation', 'test'):
            data = TokenDocuments(data_dir, split, model.base.config.vocab_size)
            # Predeclared length-only selection, independent of losses; all selected
            # articles scored in full, including the longest available tail.
            indices = sorted(range(len(data)), key=lambda i: (-data.length(i), i))[:(1 if split == 'validation' else 2)]
            conditions = {'enabled': flags}
            if split == 'test':
                conditions['fifo'] = dict(read_memory=False, read_null=False)
                if arm == 'block':
                    conditions.update(no_memory=dict(read_memory=False), zero_values=dict(zero_memory_values=True),
                        mean_only=dict(role_mask=(True, False, False)),
                        importance_only=dict(role_mask=(False, True, False)),
                        free_only=dict(role_mask=(False, False, True)))
            for condition, settings in conditions.items():
                name = split + '/' + condition
                reports[name] = evaluate(model, data, indices, name, settings, output)
                emit('condition_complete', arm=arm, condition=name, scores=reports[name])
    save_json(run / 'result.json', dict(status='completed', arm=arm, steps=steps,
        tokens=tokens_seen, cursor=cursor, reports=reports, seconds=time.monotonic() - started,
        model=manifest['model'], revision=manifest['revision'], seed=1729,
        limitations=['one seed, small frozen-backbone pilot', '512-token TBPTT credit horizon',
                     'growing summary bank; controls match physical cache, not total state',
                     'two longest test articles; fixed final checkpoint, no test-based selection']))
    emit('BLOCK_MEMORY_PILOT_COMPLETE', arm=arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--arm', choices=('gate', 'block', 'null'), required=True)
    parser.add_argument('--steps', type=int, default=128)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires Slurm GPU allocation')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    check_source(args.root)
    expected_manifest = json.loads((args.root / 'plan.json').read_text())['data_manifest_sha256']
    if sha256(args.data / 'manifest.json') != expected_manifest:
        raise ValueError('document manifest changed since submission')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    check_pretrained(manifest)
    emit('model_loading', arm=args.arm, model=manifest['model'], revision=manifest['revision'])
    if args.arm == 'gate':
        gate(args.root, args.data, manifest)
    else:
        pilot(args.root, args.data, manifest, args.arm, args.steps)


if __name__ == '__main__':
    main()
