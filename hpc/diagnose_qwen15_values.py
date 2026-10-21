"""Slurm-only diagnosis of real language-loss versus controlled sink-value gradients."""
import json
import os
from pathlib import Path
import torch
from virtualsink.uptraining.native import load_native, configure_sinks, set_window, virtual_values
from virtualsink.uptraining.qwen import parameter_groups, sink_attention
from virtualsink.uptraining.train import language_loss
from virtualsink.uptraining.data import TokenBlocks


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    root = Path(os.environ['QWEN_DIAG_ROOT'])
    data_root = Path('/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010/data')
    manifest = json.loads((data_root/'manifest.json').read_text())
    torch.manual_seed(1729)
    model = load_native(manifest['model_path'],32)
    configure_sinks(model,4,1e-9);set_window(model,512)
    parameter_groups(model,'attention',1e-4,1e-4)
    model.cuda().train();model.gradient_checkpointing_enable()
    data = TokenBlocks(data_root/'train.bin',1024,model.config.vocab_size)
    for micro in range(8):
        loss,gates = language_loss(model,data[micro][None].cuda(),32)
        (loss/8).backward()
        print('MICROBATCH',micro,'ce',loss.item(),flush=True)
        del loss,gates
    rows = []
    for i,l in enumerate(model.model.layers):
        a=l.self_attn
        real=a.sink_v.grad.abs().sum(-1).detach().cpu()
        # Moderate scores provide a guaranteed nonzero upstream signal to every value vector.
        q=torch.zeros(1,a.heads,8,a.head_dim,device='cuda')
        k,v=torch.zeros_like(q),torch.zeros_like(q)
        out,_=sink_attention(q,k,v,a.sink_w,a.sink_b,8,False,8,virtual_values(a))
        probe=torch.autograd.grad(out.sum(),a.sink_v)[0].abs().sum(-1).detach().cpu()
        row={'layer':i,'real_gradient_l1_by_head_slot':real.tolist(),
             'zero_real_head_slots':(real==0).nonzero().tolist(),
             'controlled_gradient_min':probe.min().item(),
             'controlled_all_nonzero':bool((probe>0).all()),
             'real_finite':bool(torch.isfinite(real).all())}
        rows.append(row);print(json.dumps(row),flush=True)
    assert all(r['controlled_all_nonzero'] and r['real_finite'] for r in rows)
    (root/'diagnosis.json').write_text(json.dumps(rows,indent=2))
    print('CONTROLLED_VALUE_GRADIENTS_PASSED',flush=True)


if __name__=='__main__':main()
