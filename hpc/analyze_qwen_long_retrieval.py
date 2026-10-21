"""Aggregate counterfactual probes, with block-level paired uncertainty."""
import json
import math
import random
import statistics
from pathlib import Path
from deploy_qwen_long_retrieval import LOCAL


def read_rows(path):
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(s) for s in lines if s.strip()]


def interval(values,seed=9817):
    # Independent units are eight key/background blocks, not64 independent facts.
    # Resample neutral and interfering blocks separately to preserve their mix.
    if min(values)==max(values):return [values[0],values[0]]
    rng=random.Random(seed);estimates=[]
    for _ in range(4000):
        selected=[values[rng.randrange(0,4)] for _ in range(4)]+[values[rng.randrange(4,8)] for _ in range(4)]
        estimates.append(statistics.mean(selected))
    estimates.sort()
    return [estimates[100],estimates[3899]]


def aggregate(rows):
    assert len(rows)==64 and len({(r['block'],r['truth_index']) for r in rows})==64
    groups={b:[r for r in rows if r['block']==b] for b in range(8)}
    assert all(len(g)==8 and {r['truth_index'] for r in g}==set(range(8)) for g in groups.values())
    accuracy=[statistics.mean(r['choice_correct'] for r in groups[b]) for b in range(8)]
    association=[statistics.mean(r['candidate_log_probs'][r['truth_index']]-statistics.mean(r['candidate_log_probs']) for r in groups[b]) for b in range(8)]
    spreads=[max(max(r['candidate_log_probs'][j] for r in g)-min(r['candidate_log_probs'][j] for r in g) for j in range(8)) for g in groups.values()]
    return dict(correct=sum(r['choice_correct'] for r in rows),cases=64,choice_accuracy=statistics.mean(accuracy),
        choice_accuracy_block_bootstrap95=interval(accuracy),greedy_accuracy=statistics.mean(r['greedy_correct'] for r in rows),
        choice_nll=statistics.mean(r['choice_nll'] for r in rows),vocab_nll=statistics.mean(r['vocab_nll'] for r in rows),
        counterfactual_association_log_score=statistics.mean(association),association_block_bootstrap95=interval(association),
        max_counterfactual_logprob_spread=max(spreads),block_accuracies=accuracy,
        strata={kind:dict(correct=sum(r['choice_correct'] for r in rows if r['distractors']==kind),cases=32,
            choice_accuracy=statistics.mean(r['choice_correct'] for r in rows if r['distractors']==kind),
            greedy_accuracy=statistics.mean(r['greedy_correct'] for r in rows if r['distractors']==kind))
            for kind in ('neutral_prose','other_item_records')})


def main():
    plan=json.loads((LOCAL/'plan.json').read_text());dataset=json.loads((LOCAL/'dataset.json').read_text())
    output=dict(dataset_sha256=dataset['sha256'],chance=.125,arms={},comparisons={},complete=True,
        uncertainty='Exploratory percentile bootstrap over eight key/background blocks, stratified4+4. Blocks contain counterfactual values; individual labels are not independent. Ten distances are multiple comparisons; intervals are descriptive, not confirmatory significance tests.')
    for arm in plan['arms']:
        completion=LOCAL/(arm+'-complete.json')
        if not completion.exists():
            output['complete']=False;continue
        info=json.loads(completion.read_text());rows=read_rows(LOCAL/(arm+'.jsonl'))
        assert info['status']=='complete' and info['rows']==len(rows)
        assert info['dataset_sha256']==dataset['sha256'] and info['checkpoint_step']==2048
        grouped={}
        for row in rows:
            key=(row['control'],row['filler_tokens']);grouped.setdefault(key,[]).append(row)
        stats={control:{} for control,_ in grouped}
        for (control,gap),items in sorted(grouped.items()):
            stats[control][str(gap)]=aggregate(items)
        for gap,metric in stats['missing_answer'].items():
            assert metric['correct']==8,'missing-answer labels are not balanced'
            assert abs(metric['counterfactual_association_log_score'])<1e-6
        output['arms'][arm]=dict(completion=info,metrics=stats)
    if 'memory' in output['arms']:
        memory=output['arms']['memory']['metrics']['retrieval']
        for arm,data in output['arms'].items():
            if arm=='memory':continue
            comparison={}
            for gap,metric in data['metrics']['retrieval'].items():
                deltas=[a-b for a,b in zip(memory[gap]['block_accuracies'],metric['block_accuracies'])]
                comparison[gap]=dict(memory_accuracy_minus_control=statistics.mean(deltas),paired_block_bootstrap95=interval(deltas))
            output['comparisons']['memory_vs_'+arm]=comparison
    if output['complete']:
        # A24-layer FIFO with capacity128 has no input path beyond24*127 tokens.
        for arm in ('fifo','memory_disabled'):
            for gap,metric in output['arms'][arm]['metrics']['retrieval'].items():
                if int(gap)>=4096:
                    assert metric['correct']==8 and metric['max_counterfactual_logprob_spread']<1e-4, 'FIFO finite receptive-field sanity failed'
    (LOCAL/'analysis.json').write_text(json.dumps(output,indent=2))
    print('COMPLETE',output['complete'])
    for arm,data in output['arms'].items():
        print(arm)
        for gap,metric in data['metrics']['retrieval'].items():
            print(gap,metric['correct'],metric['greedy_accuracy'],metric['counterfactual_association_log_score'])


if __name__=='__main__':main()
