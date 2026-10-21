"""Identical two-pass checkpointed training; no model/data downloads."""
import argparse
import copy
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.recurrent_sink import RecurrentSinkLM
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM
from virtualsink.uptraining.train_recurrent import token_losses


def make(cls, device):
    torch.manual_seed(732)
    cfg = dict(model_type='qwen2', vocab_size=2048, hidden_size=128, intermediate_size=384,
               num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
               hidden_act='silu', rope_theta=10000., rms_norm_eps=1e-6, tie_word_embeddings=True)
    model = cls(NativeLM(cfg, 32), capacity=128).to(device)
    model.parameter_groups('full')
    model.checkpointing = True
    # Meaningful memory contribution exercises gradients, not just tiny initial gates.
    with torch.no_grad():
        for bank in model.memory:
            bank.memory_bias.fill_(0.)
            bank.null_bias.fill_(-2.)
    return model


def update(model, optimizer, tokens, states):
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    next_states = []
    for state in states:
        losses = []
        for start in (0, 256):
            out = model(tokens[:, start:start+256], state)
            state = out.state
            losses.append(token_losses(model, out.hidden, tokens[:, start+1:start+257], 32).mean())
        loss = sum(losses)/2
        loss.backward()
        total += float(loss.detach())
        next_states.append(state.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if not torch.isfinite(norm) or not torch.isfinite(torch.tensor(total)):
        raise RuntimeError('nonfinite training')
    return next_states, total, float(norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['current', 'optimized'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires GPU Slurm job')
    torch.set_num_threads(4)
    torch.manual_seed(900)
    tokens = torch.randint(0, 2048, (1, 513), device='cuda')
    reference, optimized = make(RecurrentSinkLM, 'cuda'), make(OptimizedRecurrentSinkLM, 'cuda')
    optimized.load_state_dict(reference.state_dict())
    # End-to-end GPU parity: two passes, checkpoint backward, recurrent states, AdamW update.
    audits = []
    for model in (reference, optimized):
        opt = torch.optim.AdamW(model.parameter_groups('full', 1e-5, 3e-4))
        states, loss, norm = update(model, opt, tokens, [None, None])
        audits.append((states, loss, norm))
    torch.testing.assert_close(torch.tensor(audits[0][1:]), torch.tensor(audits[1][1:]), atol=2e-4, rtol=2e-4)
    max_state = 0.
    for sa, sb in zip(audits[0][0], audits[1][0]):
        for la, lb in zip(sa.layers, sb.layers):
            for name in ('key', 'value', 'cell', 'exposed'):
                a, b = getattr(la, name), getattr(lb, name)
                torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
                max_state = max(max_state, float((a-b).abs().max()))
    max_grad = 0.
    for (name, a), (_, b) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-3, msg=name)
            max_grad = max(max_grad, float((a.grad-b.grad).abs().max()))
    print(json.dumps({'gpu_parity': 'PASS', 'max_state_error': max_state, 'max_gradient_error': max_grad}), flush=True)
    del reference, optimized, audits, opt, model
    torch.cuda.empty_cache()
    model = make(OptimizedRecurrentSinkLM if args.arm == 'optimized' else RecurrentSinkLM, 'cuda')
    opt = torch.optim.AdamW(model.parameter_groups('full', 1e-5, 3e-4))
    states = [None, None]
    samples = []
    torch.cuda.reset_peak_memory_stats()
    # 5 warmups then 30 measured updates. Reset at identical 8192-token document boundaries.
    for step in range(35):
        if step % 16 == 0:
            states = [None, None]
        torch.cuda.synchronize()
        start = time.perf_counter()
        states, loss, norm = update(model, opt, tokens, states)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        row = dict(step=step, seconds=seconds, loss=loss, gradient_norm=norm,
                   context_position=states[0].position, measured=step >= 5)
        samples.append(row)
        print(json.dumps(row), flush=True)
    measured = [s['seconds'] for s in samples if s['measured']]
    result = dict(arm=args.arm, job_id=os.environ['SLURM_JOB_ID'], hostname=os.uname().nodename,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        config=dict(layers=4, hidden=128, intermediate=384, heads=4, kv_heads=2, vocab=2048,
                    batch=1, segment=256, segments=2, passes=2, cache=128, dtype='float32', checkpointing=True),
        gpu_parity='PASS', max_state_error=max_state, max_gradient_error=max_grad,
        mean_seconds=statistics.mean(measured), median_seconds=statistics.median(measured),
        tokens_per_second=512/statistics.mean(measured), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        samples=samples, nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,clocks.sm,temperature.gpu', '--format=csv'], universal_newlines=True))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('BENCHMARK_COMPLETE '+json.dumps({k:v for k,v in result.items() if k not in ('samples','nvidia_smi')}), flush=True)


if __name__ == '__main__':
    main()
