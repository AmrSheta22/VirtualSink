"""Full-model K80 gate: verify auxiliary gradients, valued sinks and peak-LR updates."""
import json
import os
from pathlib import Path
import time
import torch
from virtualsink.uptraining.native import load_native, set_window, configure_sinks, virtual_values
from virtualsink.uptraining.qwen import parameter_groups, auxiliary_loss
from virtualsink.uptraining.train import language_loss
from virtualsink.uptraining.data import TokenBlocks
from virtualsink.uptraining.evaluate_streaming import check_parity
from virtualsink.uptraining.sink_validation import check_value_gradients


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    root = Path(os.environ['QWEN_SEEDS_ROOT'])
    manifest = json.loads((root/'data/manifest.json').read_text())
    results = []
    for label,count,base_lr,sink_lr,auxiliary in [
        ('highlr-single',1,2e-5,5e-4,'anti_collapse'),
        ('highlr-four',4,1e-4,1e-4,'none')]:
        torch.manual_seed(1729)
        model = load_native(manifest['model_path'],32)
        configure_sinks(model,count,1e-9);set_window(model,512)
        groups = parameter_groups(model,'attention',base_lr,sink_lr)
        model.cuda().train();model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(groups,foreach=False)
        data = TokenBlocks(root/'data/train.bin',1024,model.config.vocab_size)
        tokens = data[0][None].cuda()
        started = time.monotonic();torch.cuda.reset_peak_memory_stats()
        if auxiliary != 'none':
            penalty,_ = auxiliary_loss(model.model(tokens).attentions)
            penalty.backward()
            for layer in model.model.layers:
                for param in (layer.self_attn.sink_w,layer.self_attn.sink_b,layer.self_attn.q_proj.weight):
                    assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
            del penalty,_,layer,param
            optimizer.zero_grad(set_to_none=True)
        losses, value_checks = [], []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(8):
                tokens = data[iteration*8+micro][None].cuda()
                lm,gates = language_loss(model,tokens,32)
                aux,_ = auxiliary_loss(gates,auxiliary)
                objective = (lm+.01*aux)/8
                assert torch.isfinite(objective)
                objective.backward();losses.append(lm.item())
                del objective,lm,gates,aux,_
            if count == 4:
                checks = []
                for index,layer in enumerate(model.model.layers):
                    v = layer.self_attn.sink_v
                    assert any(p is v for p in groups[1]['params'])
                    checks.append(dict(layer=index,**check_value_gradients(layer.self_attn)))
                assert sum(c['language_gradient_l1_sum'] for c in checks)>0
                value_checks.append(checks)
                print('VALUE_GRADIENT_CHECKS',json.dumps(checks),flush=True)
                before_values = [l.self_attn.sink_v.detach().clone() for l in model.model.layers]
                del layer,v
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            if count == 4:
                assert all(virtual_values(l.self_attn)[:,0].count_nonzero()==0 for l in model.model.layers)
                changes = [(l.self_attn.sink_v.detach()-before).abs().sum().item()
                           for l,before in zip(model.model.layers,before_values)]
                assert sum(changes)>0, 'AdamW must actually update the trainable values'
                print('VALUE_UPDATE_L1_BY_LAYER',json.dumps(changes),flush=True)
                del before_values
            print('SMOKE_OPTIMIZER_STEP',label,iteration+1,flush=True)
        model.eval()
        parity = check_parity(model,data[0][:641].cuda(),512)
        row = dict(label=label,optimizer_steps=2,accumulation=8,auxiliary=auxiliary,
            auxiliary_gradient_checked=auxiliary!='none',value_gradient_checked=count==4,
            value_checks=value_checks,
            parity=parity,mean_ce=sum(losses)/len(losses),seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
        results.append(row);print(json.dumps(row),flush=True)
        del optimizer,groups,model,data,tokens
        torch.cuda.empty_cache()
    (root/'smoke.json').write_text(json.dumps(results,indent=2))
    print('FULL_MODEL_GPU_CHECK_PASSED',flush=True)


if __name__ == '__main__':
    main()
