"""Matched article/position comparisons, valid for partial or completed runs."""
import json
from collections import defaultdict
from deploy_qwen_full_articles import LOCAL,ARMS


def main():
    rows={};completion={}
    for arm in ARMS:
        p=LOCAL/(arm+'.jsonl');text=p.read_text() if p.exists() else ''
        lines=text.splitlines()
        if text and not text.endswith('\n'):lines=lines[:-1]
        rr=[json.loads(line) for line in lines]
        rows[arm]={(r['split'],r['document'],r['start']):r for r in rr}
        assert len(rows[arm])==len(rr),'Duplicate article positions'
        p=LOCAL/(arm+'-complete.json')
        completion[arm]=json.loads(p.read_text()) if p.exists() else None
        if completion[arm]:
            assert completion[arm]['rows']==len(rr)
            assert completion[arm]['tokens']==sum(r['tokens'] for r in rr)
            expected=json.loads((LOCAL/'data-coverage.json').read_text())
            assert completion[arm]['tokens']==sum(s['predictions'] for s in expected.values())
    matched=set.intersection(*(set(rows[arm]) for arm in ARMS))
    output=dict(complete=all(completion.values()),completion=completion,matched_segments=len(matched),splits={})
    for split in ('test','validation'):
        bands={}
        for band in ('0-4096','4096-8192','8192-12288','12288-16384'):
            perdoc={arm:defaultdict(lambda:[0.,0]) for arm in ARMS}
            for key in sorted(matched):
                if key[0]!=split:continue
                for arm in ARMS:
                    value=rows[arm][key]['bands'][band]
                    if value['tokens']:
                        perdoc[arm][key[1]][0]+=value['loss_sum']
                        perdoc[arm][key[1]][1]+=value['tokens']
            metrics={}
            for arm in ARMS:
                docs=perdoc[arm];count=sum(v[1] for v in docs.values())
                metrics[arm]=dict(documents=len(docs),tokens=count,
                    ce=sum(v[0] for v in docs.values())/count if count else None,
                    per_document={str(d):dict(tokens=v[1],ce=v[0]/v[1]) for d,v in docs.items()})
            deltas={}
            for control in ('memory_disabled','fifo'):
                differences={d:metrics['memory']['per_document'][d]['ce']-v['ce']
                    for d,v in metrics[control]['per_document'].items()}
                deltas[control]=dict(per_article_ce_difference=differences,
                    mean_article_difference=sum(differences.values())/len(differences) if differences else None,
                    articles_memory_lower_ce=sum(v<0 for v in differences.values()))
            bands[band]=dict(arms=metrics,memory_minus_control=deltas)
        output['splits'][split]=bands
    (LOCAL/'analysis.json').write_text(json.dumps(output,indent=2))
    lines=['# Full-article evaluation','',f"Complete: {output['complete']}. Comparisons use only identical article segments finished by all three arms.",'',
        'Fixed validation-selected step2048 checkpoints. Full articles, no new training or selection. Lower CE is better. Different position ranges contain different articles; per-article paired differences are retained in analysis.json. This is a post-hoc extension of the previously inspected test set.','',
        '| Split | Positions | Articles | Matched tokens | Memory CE | Memory disabled CE | FIFO CE |','|---|---|---:|---:|---:|---:|---:|']
    for split,bands in output['splits'].items():
        for band,stats in bands.items():
            a=stats['arms'];m=a['memory']
            fmt=lambda v:f'{v:.6f}' if v is not None else 'pending'
            lines.append(f"| {split} | {band} | {m['documents']} | {m['tokens']} | "+' | '.join(fmt(a[arm]['ce']) for arm in ARMS)+' |')
    (LOCAL/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
