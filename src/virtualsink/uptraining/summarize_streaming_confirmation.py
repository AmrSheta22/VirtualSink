"""Document-paired confirmation report; no training or GPU required."""
import argparse
import html
import json
import math
from pathlib import Path
import numpy as np
from .evaluate_streaming import CONFIRMATION_TITLES, write


def paired_effect(standard, sink):
    standard=np.asarray(standard,dtype=float); sink=np.asarray(sink,dtype=float)
    if standard.shape!=sink.shape or standard.ndim!=1 or len(standard)<2:
        raise ValueError('paired article arrays required')
    delta=standard-sink
    if not np.isfinite(delta).all(): raise ValueError('nonfinite loss')
    rng=np.random.default_rng(1729)
    samples=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
    low,high=np.quantile(samples,[.025,.975])
    return {'articles':len(delta),'mean_ce_advantage':float(delta.mean()),
            'article_bootstrap_95pct_ce_interval':[float(low),float(high)],
            'sink_to_standard_ppl_ratio':math.exp(-float(delta.mean())),
            'ppl_ratio_95pct_interval':[math.exp(-float(high)),math.exp(-float(low))],
            'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),
            'worst_article_ce_advantage':float(delta.min()),
            'conclusion':('supported advantage on this confirmation set' if low>0 else
                          'supported disadvantage on this confirmation set' if high<0 else
                          'inconclusive on this confirmation set')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    rows=[];missing=[];identities={}
    for model in ['qwen05','smollm135']:
        for index,title in enumerate(CONFIRMATION_TITLES):
            p=root/'results'/model/f'article{index:02d}'
            if not (p/'complete.json').exists():
                missing.append(str(p));continue
            config=json.loads((p/'config.json').read_text())
            assert config['articles'][0]['title']==title and config['confirmation_index']==index
            identity={k:config[k] for k in ['checkpoint_sha256','training_config','source_sha256','position_policy','window','predictions_per_article']}
            if model in identities: assert identity==identities[model],'inconsistent experiment settings'
            identities[model]=identity
            comp=json.loads((p/'comparisons.json').read_text())['0']
            for policy in ['standard-fifo','standard-prefix4','query_sink-fifo','query_sink-disabled']:
                result=json.loads((p/f'{policy}-article0.json').read_text())
                assert result['peak_cache_entries']==512
                assert len(result['per_token_ce'])==8192
                row={'model':model,'article_index':index,'title':title,'policy':policy,
                     **comp[f'{policy}-article0']}
                row['virtual_before']=float(np.mean(result['before_mean_virtual_by_layer_head']))
                row['virtual_after']=float(np.mean(result['after_mean_virtual_by_layer_head']))
                rows.append(row)
    write(root/'article_results.json',rows)
    if missing:
        write(root/'INCOMPLETE.json',{'missing':missing,'completed_article_model_pairs':len(rows)//4})
        (root/'REPORT.md').write_text('# Streaming confirmation incomplete\n\nNo final conclusion: '+str(len(missing))+' article/model units missing. See INCOMPLETE.json.\n')
        raise SystemExit(2)
    report=['# Streaming confirmation: 12 new held-out articles','',
      'The original two exploratory articles are excluded from every statistic below. All 12 remaining articles in the official WikiText test split with at least 8193 tokens under BOTH tokenizers were selected before observing results. Each model scores 8192 predictions per article with a physical cache capacity of512.','',
      'These are the completed regularized SWA step1000 standard/sink checkpoints. Absolute RoPE, no rebasing. The prefix control is not an exact StreamingLLM reproduction.','',
      'Primary endpoint: mean paired standard-FIFO-minus-virtual-FIFO cross-entropy over input positions512–8191. Positive favors virtual sinks. Bootstrap resamples articles (10000 draws, seed1729), not tokens.','']
    stats={}
    for model in ['qwen05','smollm135']:
        data=[r for r in rows if r['model']==model]
        values=lambda policy,region: [r[region]['ce'] for r in data if r['policy']==policy]
        stats[model]={}
        report += [f'## {model}','', '| Condition | Post-eviction PPL | PPL beyond10w |','|---|---:|---:|']
        for policy in ['standard-fifo','standard-prefix4','query_sink-fifo','query_sink-disabled']:
            report.append(f'| {policy} | {math.exp(np.mean(values(policy,"post_eviction"))):.4f} | {math.exp(np.mean(values(policy,"beyond_10w"))):.4f} |')
        for region in ['post_eviction','beyond_10w','pre_eviction']:
            stats[model][region]=paired_effect(values('standard-fifo',region),values('query_sink-fifo',region))
        stats[model]['disabled_vs_enabled']=paired_effect(values('query_sink-disabled','post_eviction'),values('query_sink-fifo','post_eviction'))
        stats[model]['prefix_vs_virtual']=paired_effect(values('standard-prefix4','post_eviction'),values('query_sink-fifo','post_eviction'))
        s=stats[model]['post_eviction'];low,high=s['article_bootstrap_95pct_ce_interval']
        report += ['',f'Primary conclusion: **{s["conclusion"]}**. Wins: {s["wins"]}/12. Mean CE advantage {s["mean_ce_advantage"]:.6f}, 95% article-bootstrap interval [{low:.6f}, {high:.6f}]. Sink/standard perplexity ratio {s["sink_to_standard_ppl_ratio"]:.6f}.','',
          '| New article | Standard FIFO CE | Virtual FIFO CE | CE advantage |','|---|---:|---:|---:|']
        for i,title in enumerate(CONFIRMATION_TITLES):
            st=next(r['post_eviction']['ce'] for r in data if r['article_index']==i and r['policy']=='standard-fifo')
            si=next(r['post_eviction']['ce'] for r in data if r['article_index']==i and r['policy']=='query_sink-fifo')
            report.append(f'| {title.strip("= ")} | {st:.6f} | {si:.6f} | {st-si:.6f} |')
        d=[r for r in data if r['policy']=='query_sink-fifo']
        report += ['',f'Mean virtual mass before/after eviction: {100*np.mean([r["virtual_before"] for r in d]):.4f}% / {100*np.mean([r["virtual_after"] for r in d]):.4f}%.','']
    write(root/'statistics.json',stats)
    report += ['## Interpretation limits','',
      'This establishes evidence only for these checkpoints, this corpus, this cache budget and positional policy. It does not establish general superiority over full attention or StreamingLLM, cross-seed repeatability, or the effect of removing auxiliary loss. Wiki articles can share topics, limiting article independence. The beyond10w, prefix and disabled comparisons are secondary, not separately corrected confirmatory endpoints. Confidence intervals do not measure training-seed uncertainty. A nonzero mean alone is not enough: read its interval, wins and worst article.','',
      'No perfectly flat loss, NaN, or fixed spike is required for success. More contexts and training seeds remain necessary for broader claims.','']
    (root/'REPORT.md').write_text('\n'.join(report))
    write(root/'complete.json',{'complete':True,'new_articles':12,'models':2,'policies':4,'predictions':12*2*4*8192})
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
