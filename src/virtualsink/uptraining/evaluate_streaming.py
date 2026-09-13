"""Frozen paired-model FIFO eviction evaluation on individual held-out articles."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from .native import load_native, set_window
from .qwen import disable_sinks
from .data import sha256
from .streaming_fifo import StreamingDecoder

CONFIRMATION_TITLES = [
    '= Ise @-@ class battleship =', '= Second Battle of Naktong Bulge =',
    '= Ironclad warship =', '= Temnospondyli =', '= Clayton Kershaw =',
    '= Battle of D\u00fcrenstein =', '= Brock Lesnar =', '= Olmec colossal heads =',
    '= Brad Stevens =', '= Nero =', '= Manila =', '= Bintulu =',
]


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def heatmap(path, values, title):
    rows, cols = values.shape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{80+cols*40}" height="{80+rows*20}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="10" y="20">{html.escape(title)}; white=0 blue=1</text>']
    for l in range(rows):
        svg.append(f'<text x="5" y="{55+l*20}">L{l}</text>')
        for h in range(cols):
            c = int(255*(1-float(values[l,h])))
            svg.append(f'<rect x="{50+h*40}" y="{40+l*20}" width="39" height="19" fill="rgb({c},{c},255)"><title>L{l} H{h}: {values[l,h]:.6f}</title></rect>')
    path.write_text('\n'.join(svg+['</svg>']))


def curves(out, results):
    colors=['#2563eb','#16a34a','#dc2626','#9333ea']
    for sid in range(2):
        selected={k:v for k,v in results.items() if k.endswith(f'article{sid}')}
        if not selected: continue
        ymax=max(b['ce'] for v in selected.values() for b in v['bins'])*1.1
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="60" y="25">Article {sid}: cross-entropy in 128-token bins</text>']
        for j,(label,result) in enumerate(selected.items()):
            color=colors[j%4]
            points=' '.join(f'{60+900*(b["input_start"]+64)/8192},{430-370*b["ce"]/ymax}' for b in result['bins'])
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            svg.append(f'<text x="60" y="{480+22*j}" fill="{color}">{label}</text>')
        for pos in [0,512,1024,2560,5120,8192]:
            x=60+900*pos/8192
            svg.append(f'<path d="M{x} 50 V430" stroke="#bbb" stroke-dasharray="4 4"/><text x="{x}" y="450">{pos}</text>')
        for fraction in [0,.25,.5,.75,1]:
            svg.append(f'<text x="5" y="{430-370*fraction}">{ymax*fraction:.2f}</text>')
        (out/f'article{sid}-loss.svg').write_text('\n'.join(svg+['</svg>']))


@torch.no_grad()
def check_parity(model, tokens, window):
    set_window(model, window)
    hidden = model.model(tokens[None]).last_hidden_state[0]
    decoder = StreamingDecoder(model, window)
    maximum = 0.
    for i, token in enumerate(tokens):
        actual, *_ = decoder.step(token)
        reference = model.lm_head(hidden[i])
        error = (actual-reference).abs().max().item()
        maximum = max(maximum,error)
        torch.testing.assert_close(actual, reference, atol=.003, rtol=.0001)
        assert all(k.shape[2] <= (window or len(tokens)) for k,v in decoder.cache)
    return {'tokens':len(tokens),'window':window,'max_absolute_logit_error':maximum,
            'atol':.003,'rtol':.0001}


@torch.no_grad()
def evaluate(model, ids, tokenizer, prefix, out, label):
    decoder = StreamingDecoder(model, 512, prefix)
    losses, masses, scores, lses = [], [], [], []
    started = time.monotonic()
    for i, token in enumerate(ids[:-1]):
        logits,a,s,l = decoder.step(token)
        loss = torch.nn.functional.cross_entropy(logits[None],ids[i+1:i+2]).item()
        if not math.isfinite(loss):
            write(out/f'{label}-failure.json',{'position':i,'reason':'nonfinite loss'})
            raise FloatingPointError(f'nonfinite loss at {i}')
        assert all(k.shape[2]<=512 for k,v in decoder.cache)
        losses.append(loss); masses.append(a.cpu().numpy())
        scores.append(s.cpu().numpy()); lses.append(l.cpu().numpy())
        if (i+1)%512==0:
            write(out/f'{label}-progress.json',{'positions':i+1,'elapsed_seconds':time.monotonic()-started,
                  'seconds_per_token':(time.monotonic()-started)/(i+1),'last512_ce':float(np.mean(losses[-512:]))})
            print(label,i+1,'CE',np.mean(losses[-512:]),flush=True)
    a=np.array(masses); s=np.array(scores); l=np.array(lses)
    np.savez_compressed(out/f'{label}-diagnostics.npz',virtual_mass=a,sink_score=s,physical_lse=l,
                        input_token_ids=ids[:-1].cpu().numpy(),per_token_ce=np.array(losses))
    bins=[]
    for start in range(0,len(losses),128):
        ce=float(np.mean(losses[start:start+128]))
        bins.append({'input_start':start,'input_end_exclusive':min(start+128,len(losses)),
                     'ce':ce,'perplexity':math.exp(min(ce,700)),
                     'virtual_mass':float(a[start:start+128].mean())})
    examples=[]
    for layer in range(a.shape[1]):
        for head in range(a.shape[2]):
            for name,pos in [('high',int(a[:,layer,head].argmax())),('low',int(a[:,layer,head].argmin()))]:
                examples.append({'layer':layer,'head':head,'kind':name,'position':pos,
                  'token':tokenizer.decode([int(ids[pos])]),
                  'context':tokenizer.decode(ids[max(0,pos-8):pos+1].tolist()),
                  'mass':float(a[pos,layer,head])})
    result={'bins':bins,'per_token_ce':losses,'mean_ce':float(np.mean(losses)),
            'prefix':prefix,'capacity':512,'peak_cache_entries':max(k.shape[2] for k,v in decoder.cache),
            'elapsed_seconds':time.monotonic()-started,'semantic_examples':examples}
    for name,left,right in [('before',0,512),('after',512,len(a))]:
        means=a[left:right].mean(0)
        result[name+'_mean_virtual_by_layer_head']=means.tolist()
        heatmap(out/f'{label}-{name}-heads.svg',means,label+' '+name)
    write(out/f'{label}.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['qwen05','smollm135'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirmation-index',type=int,choices=range(len(CONFIRMATION_TITLES)))
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        parser.error('Slurm GPU allocation required')
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    root=Path.home()/'data/phase3-swa-20260910/runs'
    runs={v:root/f'{args.model}-{v}' for v in ['standard','query_sink']}
    cfg={v:json.loads((p/'config.json').read_text()) for v,p in runs.items()}
    for key in ['manifest','mode','seq_len','window_size','steps','batch_size','accumulation','seed']:
        assert cfg['standard'][key]==cfg['query_sink'][key],key
    for p in runs.values(): assert (p/'summary.json').exists()
    manifest=cfg['standard']['manifest']
    base=Path.home()/'data/phase3-k80-20260909/hf-cache/hub'
    source=next(base.glob('datasets--Salesforce--wikitext/snapshots/'+manifest['source']['revision']+'/wikitext-103-raw-v1/test-*.parquet'))
    assert sha256(source)==manifest['splits']['test']['source_files'][0]['sha256']
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(manifest['model_path'])/'tokenizer.json'))
    docs={}; title=None; parts=[]
    for row in pq.read_table(source,columns=['text']).column('text').to_pylist():
        if re.match(r'^\s*= [^=].*[^=] =\s*$',row):
            if title: docs[title]=''.join(parts)
            title=row.strip(); parts=[row]
        elif title is not None: parts.append(row)
    if title: docs[title]=''.join(parts)
    selected=['= American Beauty ( 1999 film ) =','= 2003 Pacific typhoon season =']
    if args.confirmation_index is not None:
        selected=[CONFIRMATION_TITLES[args.confirmation_index]]
    texts=[docs[t] for t in selected]
    encoded=[tokenizer.encode(t).ids for t in texts]
    assert all(len(t)>=8193 for t in encoded)
    identity={'model':args.model,'job':os.environ['SLURM_JOB_ID'],'window':512,'predictions_per_article':8192,
       'confirmation_index':args.confirmation_index,
       'articles':[{'title':t,'text_sha256':hashlib.sha256(s.encode()).hexdigest(),'available_tokens':len(ids)} for t,s,ids in zip(selected,texts,encoded)],
       'checkpoint_sha256':{v:sha256(p/'best.pt') for v,p in runs.items()},'training_config':cfg,
       'position_policy':'absolute RoPE; no rebasing; prefix control is not exact StreamingLLM positional treatment',
       'scope':'held-out individual WikiText articles, first 8192 next-token targets; teacher-forced; no cache resets within article',
       'eviction':'512 physical entries including current token; first eviction at zero-based input 512 predicting target 513',
       'source_sha256':{n:sha256(Path(__file__).parent/n) for n in ['evaluate_streaming.py','streaming_fifo.py','native.py','qwen.py']}}
    write(out/'config.json',identity)
    results={}
    for variant in ['standard','query_sink']:
        torch.manual_seed(cfg[variant]['seed'])
        model=load_native(manifest['model_path'],64).eval()
        initial={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith(('sink_w','sink_b'))}
        saved=torch.load(runs[variant]/'best.pt',map_location='cpu')
        assert saved['step']==1000
        model.load_state_dict(saved['model']); del saved
        if variant=='query_sink':
            params={n:{'initial':initial[n].tolist(),'trained':p.detach().tolist(),
                   'initial_norm_by_head':initial[n].norm(dim=-1).tolist() if n.endswith('sink_w') else None,
                   'trained_norm_by_head':p.detach().norm(dim=-1).tolist() if n.endswith('sink_w') else None}
                   for n,p in model.named_parameters() if n in initial}
            write(out/'sink_parameters.json',params)
        model.cuda(); disable_sinks(model,variant=='standard')
        test=torch.tensor(encoded[0][:641],device='cuda')
        checks={str(w):check_parity(model,test,w) for w in [None,512]}
        write(out/f'parity-{variant}.json',checks)
        policies=[('fifo',0,variant=='standard')]
        policies += [('prefix4',4,True)] if variant=='standard' else [('disabled',0,True)]
        for sid,ids in enumerate(encoded):
            tokens=torch.tensor(ids[:8193],device='cuda')
            for policy,prefix,disabled in policies:
                disable_sinks(model,disabled)
                label=f'{variant}-{policy}-article{sid}'
                evaluate(model,tokens,tokenizer,prefix,out,label)
                results[label]=json.loads((out/f'{label}.json').read_text())
                write(out/'summary.json',{k:{'mean_ce':v['mean_ce'],'bins':v['bins']} for k,v in results.items()})
                curves(out,results)
        del model
        torch.cuda.empty_cache()
    comparisons={}
    for sid in range(len(encoded)):
        reference=np.array(results[f'standard-prefix4-article{sid}']['per_token_ce'])
        comparisons[str(sid)]={}
        for label,result in results.items():
            if label.endswith(f'article{sid}'):
                ce=np.array(result['per_token_ce'])
                comparisons[str(sid)][label]={name:{'ce':float(ce[a:b].mean()),
                    'excess_ce_vs_standard_prefix4':float((ce-reference)[a:b].mean())}
                    for name,a,b in [('pre_eviction',0,512),('post_eviction',512,8192),('beyond_10w',5120,8192)]}
    write(out/'comparisons.json',comparisons)
    write(out/'complete.json',{'complete':True,'conditions_per_article':4,'articles':len(encoded)})


if __name__=='__main__': main()
