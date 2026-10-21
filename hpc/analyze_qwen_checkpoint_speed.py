"""Local comparison excluding diagnostic-contaminated matched updates."""
import json
import statistics
from deploy_qwen_checkpoint_speed import LOCAL, ROOT


def main():
    if not (LOCAL/'comparison.json').exists():
        print('WAITING_FOR_PARITY_AND_COMPLETION')
        return
    raw=json.loads((LOCAL/'comparison.json').read_text())
    assert raw['parity']=='PASS'
    current,optimized=[json.loads((LOCAL/(arm+'.json')).read_text()) for arm in ('current','optimized')]
    policy=json.loads((LOCAL/'timing-exclusions.json').read_text())
    cs={r['step']:r for r in current['samples']}
    os={r['step']:r for r in optimized['samples']}
    steps=policy['primary_matched_steps']
    for step in steps:
        assert cs[step]['tokens']==os[step]['tokens'] and cs[step]['cursor']==os[step]['cursor']
    a=sum(cs[s]['seconds'] for s in steps)
    b=sum(os[s]['seconds'] for s in steps)
    n=sum(cs[s]['tokens'] for s in steps)
    full=[s for s in steps if cs[s]['tokens']==512]
    history=[json.loads(s) for s in (ROOT/'artifacts/qwen05-long-uptraining-20261014-r1/runs/ordinary/metrics.jsonl').read_text().splitlines()]
    history={r['step']:r for r in history if r['event']=='train'}
    replay=[s for s in cs if s in history]
    result=dict(checkpoint_step=raw['checkpoint_step'],job=raw['job'],
        matched_steps=steps, excluded_matched_steps=policy['excluded_matched_steps'],
        current_mean_seconds=a/len(steps), optimized_mean_seconds=b/len(steps),
        speedup=a/b,time_reduction_percent=100*(1-b/a),
        current_tokens_per_second=n/a, optimized_tokens_per_second=n/b,
        per_update_speedups={s:cs[s]['seconds']/os[s]['seconds'] for s in steps},
        full_512_token_steps=full,
        full_512_current_seconds=statistics.mean(cs[s]['seconds'] for s in full),
        full_512_optimized_seconds=statistics.mean(os[s]['seconds'] for s in full),
        full_512_speedup=sum(cs[s]['seconds'] for s in full)/sum(os[s]['seconds'] for s in full),
        current_peak_allocated_gb=raw['current_peak_allocated_gb'],optimized_peak_allocated_gb=raw['optimized_peak_allocated_gb'],
        parity=raw['parity'], gradient_parity=raw['gradient_parity'], max_state_difference=raw['max_state_difference'],
        max_loss_difference=raw['max_loss_difference'],
        reference_replay_steps=replay,
        max_reference_loss_difference_from_production=max(abs(cs[s]['losses'][0]-history[s]['ce']) for s in replay),
        max_reference_gradient_norm_difference_from_production=max(abs(cs[s]['gradient_norm']-history[s]['gradient_norm']) for s in replay))
    (LOCAL/'audited-comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
