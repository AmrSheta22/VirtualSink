"""Evaluate both new selected checkpoints against the original standard control."""
import argparse
import json
import os
from pathlib import Path
import torch
from .native import load_native, configure_sinks, virtual_values
from .qwen import disable_sinks
from .data import sha256
from .qwen15_streaming import article, rebase_parity
from .evaluate_streaming import check_parity, evaluate, write

ORIGINAL = Path('/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010/seeds/seed1729/runs/standard')
POLICIES = ['standard-fifo', 'standard-prefix4', 'standard-rebased-prefix4',
            'highlr-single-fifo', 'highlr-single-disabled', 'highlr-four-fifo', 'highlr-four-disabled']


def runs(root):
    return {'standard': ORIGINAL, 'highlr-single': root/'runs/highlr-single', 'highlr-four': root/'runs/highlr-four'}


def pin(root):
    identity = {}
    for label, run in runs(root).items():
        cfg = json.loads((run/'config.json').read_text())
        summary = json.loads((run/'summary.json').read_text())
        assert cfg['steps'] == 1000 and cfg['seed'] == 1729
        identity[label] = {'sha256': sha256(run/'best.pt'), 'best_step': summary['best_step'], 'config': cfg}
    assert all(v['config']['manifest'] == identity['standard']['config']['manifest'] for v in identity.values())
    write(root/'checkpoint-identity.json',identity)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--article-index',type=int,choices=range(12),default=0)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--pin',action='store_true')
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if a.pin:
        pin(a.root); return
    assert torch.cuda.is_available()
    out = a.root/('validation/unit' if a.validate_only else f'results/article{a.article_index:02d}')
    out.mkdir(parents=True,exist_ok=False)
    identity = json.loads((a.root/'checkpoint-identity.json').read_text())
    tokenizer, encoded, doc = article(identity['standard']['config']['manifest'],a.article_index)
    tokens = torch.tensor(encoded,device='cuda')
    write(out/'config.json',{'article_index':a.article_index,'article':doc,'identity':identity,
        'policies':POLICIES,'capacity':512,'predictions':8192,
        'source_sha256':{f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
        'job':os.environ['SLURM_JOB_ID'],'sink_score_semantics':'logsumexp of all virtual logits; mass is total virtual probability'})
    for label, run in runs(a.root).items():
        info = identity[label]; cfg = info['config']
        assert sha256(run/'best.pt') == info['sha256']
        torch.manual_seed(cfg['seed'])
        model = load_native(cfg['manifest']['model_path'],32)
        configure_sinks(model,cfg.get('sink_count',1),cfg.get('sink_value_init_scale',1e-9))
        saved = torch.load(run/'best.pt',map_location='cpu')
        assert saved['step'] == info['best_step'] and saved['config'] == cfg
        model.load_state_dict(saved['model']); del saved
        model.cuda().eval(); disable_sinks(model,label == 'standard')
        checks = {str(w):check_parity(model,tokens[:641],w) for w in [None,512]}
        if label == 'standard':
            checks['rebased'] = rebase_parity(model,tokens[:1153] if a.validate_only else tokens[:641])
        write(out/f'parity-{label}.json',checks)
        if label == 'highlr-four':
            write(out/'sink-values.json',{str(i):{'trainable_value_rms_by_head':l.self_attn.sink_v.square().mean(-1).sqrt().tolist(),
                'zero_slot_max_abs':virtual_values(l.self_attn)[:,0].abs().max().item()} for i,l in enumerate(model.model.layers)})
        print('PARITY_PASSED',label,flush=True)
        if not a.validate_only:
            policies = [('standard-fifo',0,False,True),('standard-prefix4',4,False,True),('standard-rebased-prefix4',4,True,True)] if label == 'standard' else [
                (label+'-fifo',0,False,False),(label+'-disabled',0,False,True)]
            for policy,prefix,rebase,disabled in policies:
                disable_sinks(model,disabled)
                evaluate(model,tokens,tokenizer,prefix,out,policy,rebase=rebase)
        del model; torch.cuda.empty_cache()
    write(out/'complete.json',{'complete':True,'article_index':a.article_index,'validate_only':a.validate_only})


if __name__ == '__main__':
    main()
