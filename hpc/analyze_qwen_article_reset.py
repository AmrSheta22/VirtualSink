"""Audit reset interventions and compare identical target tokens."""
import hashlib,json,math,tarfile
from collections import defaultdict
from deploy_qwen_article_reset import LOCAL,ROOT


def main():
    complete=json.loads((LOCAL/'complete.json').read_text());assert complete['status']=='complete'
    rows=[json.loads(s) for s in (LOCAL/'scores.jsonl').read_text().splitlines()]
    groups=defaultdict(dict)
    for r in rows:
        g=groups[r['condition']]
        for i,loss in enumerate(r['losses']):
            pos=r['start']+i;assert pos not in g and math.isfinite(loss);g[pos]=loss
    assert sum(len(g) for g in groups.values())==complete['scored_predictions']
    base=groups['intact'];assert sorted(base)==list(range(13213))
    old=ROOT/'artifacts/qwen05-full-article-evaluation-20261017-r1'
    original={r['start']:r for r in map(json.loads,(old/'memory.jsonl').read_text().splitlines()) if r['split']=='test' and r['document']==25}
    max_delta=max(abs(sum(base[i] for i in range(start,start+r['tokens']))-r['loss_sum'])/r['tokens'] for start,r in original.items())
    assert max_delta<1e-6,'Intact run differs from earlier baseline'
    with tarfile.open(LOCAL/'release.tar.gz') as t:
        for n,h in json.loads((LOCAL/'source-sha256.json').read_text()).items():
            assert hashlib.sha256(t.extractfile(n).read()).hexdigest()==h
    assert all(p.stat().st_size==0 for p in (LOCAL/'logs').glob('*.err'))
    def ce(g,lo,hi):return sum(g[i] for i in range(lo,hi))/(hi-lo)
    intact=ce(base,12288,13213);results={}
    for condition,g in groups.items():
        if condition=='intact':continue
        boundary=int(condition.split('_')[1]);assert sorted(g)==list(range(boundary,13213))
        merged={**base,**g}
        results[condition]=dict(boundary=boundary,final925_ce=ce(merged,12288,13213),
            final925_delta_vs_intact=ce(merged,12288,13213)-intact,
            post_reset_ce=ce(g,boundary,13213),intact_same_suffix_ce=ce(base,boundary,13213),
            tail_token_losses_lower=sum(g[i]<base[i] for i in range(max(boundary,12288),13213)),
            affected_tail_tokens=13213-max(boundary,12288),
            blocks={str(start):dict(tokens=min(start+256,13213)-start,
                reset_ce=ce(merged,start,min(start+256,13213)),
                intact_ce=ce(base,start,min(start+256,13213))) for start in range(12288,13213,256)})
    output=dict(status='complete',intact_tail_ce=intact,earlier_baseline_max_segment_ce_difference=max_delta,
        results=results,completion=complete)
    (LOCAL/'analysis.json').write_text(json.dumps(output,indent=2))
    lines=['# Brock Lesnar article: recurrent-memory reset','',
        'Exploratory single-article intervention on fixed step2048 weights. Physical K/V and absolute position are preserved exactly at each reset; all recurrent cells and exposed states are cleared once, then normal writes resume. Original snapshots remain unchanged.','',
        f'Uninterrupted CE on the final925 predictions: {intact:.6f}. Earlier baseline reproduced within {max_delta:.2g} CE per segment.','',
        '| Reset at input position | Tokens to12K scoring boundary | Final925 CE | Increase vs intact |','|---:|---:|---:|---:|']
    for r in results.values():lines.append(f"| {r['boundary']} | {12288-r['boundary']} | {r['final925_ce']:.6f} | {r['final925_delta_vs_intact']:+.6f} |")
    lines+=['','Reset12544 happens256 tokens into the final925; its first256 targets use unchanged intact state. Negative distance in that row denotes an intervention after scoring begins. This is not a full925-token reset exposure.','',
        'Zeroing state changes its magnitude and familiarity as well as removing content. Preserved physical caches may already carry earlier memory information. A reset effect establishes dependence on the accumulated state under this intervention, but does not identify the age or semantic content of the useful information. Article and boundaries were chosen after inspecting earlier gains; this is not independent confirmation.','',
        'Per-token losses and per256-token block comparisons are retained in scores.jsonl and analysis.json. No training or checkpoint selection occurred.']
    (LOCAL/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
