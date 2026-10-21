"""Require twelve complete article units; compare new sink runs with standard control."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .qwen15_ablation_streaming import POLICIES
from .evaluate_streaming import CONFIRMATION_TITLES,write
from .summarize_qwen15_streaming import token_scores,REGIONS
from .summarize_streaming_confirmation import paired_effect


def summarize(root):
    rows,missing,common = [],[],None
    identity_path = root/'checkpoint-identity.json'
    if not identity_path.exists():
        write(root/'INCOMPLETE.json',{'missing':['checkpoint-identity.json'],
                                    'reason':'Training or checkpoint pinning did not complete; no streaming conclusion.'})
        raise ValueError('streaming checkpoint identity unavailable; see INCOMPLETE.json')
    identity = json.loads(identity_path.read_text())
    for i,title in enumerate(CONFIRMATION_TITLES):
        folder = root/'results'/f'article{i:02d}'
        if not (folder/'complete.json').exists():
            missing.append(str(folder));continue
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['article_index']==i and cfg['article']['title']==title
        assert cfg['identity']==identity and cfg['policies']==POLICIES
        assert cfg['capacity']==512 and cfg['predictions']==8192
        if common is None:
            common = cfg['source_sha256']
        assert common==cfg['source_sha256']
        ids = None
        for policy in POLICIES:
            result = json.loads((folder/f'{policy}.json').read_text())
            assert result['capacity']==result['peak_cache_entries']==512
            assert result['prefix']==(4 if 'prefix4' in policy else 0)
            assert result['rebase']==(policy=='standard-rebased-prefix4')
            with np.load(folder/f'{policy}-diagnostics.npz') as d:
                np.testing.assert_array_equal(d['per_token_ce'],result['per_token_ce'])
                if ids is None:
                    ids = d['input_token_ids'].copy()
                np.testing.assert_array_equal(ids,d['input_token_ids'])
            rows.append({'article_index':i,'title':title,'policy':policy,**token_scores(result)})
    write(root/'article_results.json',rows)
    if missing:
        write(root/'INCOMPLETE.json',{'missing':missing})
        raise ValueError('incomplete streaming evaluation')
    values = lambda p,r:[row[r] for row in rows if row['policy']==p]
    metrics = {p:{r:{'ce':float(np.mean(values(p,r))),
                     'perplexity':math.exp(float(np.mean(values(p,r))))} for r in REGIONS} for p in POLICIES}
    effects = {run:{control:paired_effect(values(control,'post_eviction'),values(run+'-fifo','post_eviction'))
               for control in ['standard-fifo','standard-prefix4','standard-rebased-prefix4',run+'-disabled']}
               for run in ['highlr-single','highlr-four']}
    effects['four_vs_single'] = paired_effect(values('highlr-single-fifo','post_eviction'),values('highlr-four-fifo','post_eviction'))
    write(root/'statistics.json',{'metrics':metrics,'paired_effects':effects,'identity':identity,
        'scope':'One seed; reused twelve held-out articles. Runs differ in architecture, LR schedule and auxiliary objective; comparisons do not isolate one factor.'})
    lines = ['# Qwen1.5B sink training experiments','','| Policy | Post-eviction PPL | Beyond 10 windows PPL |','|---|---:|---:|']
    lines += [f"| {p} | {metrics[p]['post_eviction']['perplexity']:.4f} | {metrics[p]['beyond_10w']['perplexity']:.4f} |" for p in POLICIES]
    lines += ['','Paired article bootstrap (10000 resamples):','',json.dumps(effects,indent=2),'',
        'One training seed and reused article set. Different objectives and schedules prevent attributing differences solely to sink count. Packed trainer tests and persistent streaming results are distinct.']
    (root/'REPORT.md').write_text('\n'.join(lines))
    write(root/'complete.json',{'complete':True,'articles':12,'policies':7})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
