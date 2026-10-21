"""Complete the revised screen, explicitly excluding user-cancelled recomputation."""
import json,math,os
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(os.environ['PILOT_STREAM_ROOT']);old=Path(os.environ['PILOT_REFERENCE_ROOT'])
    result={'pretrained':{},'trained':{},'missing':[],
            'intentional_cancellations':['217043_2 Pythia recomputation','217043_5 Qwen recomputation'],
            'scope':'400k PG19 predictions per policy; matched final256-step Pythia410M checkpoints; one seed. No recomputation comparison. Different tokenizers prohibit cross-model PPL ranking.'}
    for group,source,labels in [('pretrained',old,['pythia28','qwen15']),('trained',root,['standard','legacy','calibrated'])]:
        for label in labels:
            result[group][label]={}
            for policy in ['fifo','streamingllm']:
                folder=source/'results'/(label+'-'+policy)
                if not (folder/'complete.json').exists():result['missing'].append(str(folder));continue
                losses=np.load(folder/'losses.npy');complete=json.loads((folder/'complete.json').read_text())
                assert losses.shape==(400000,) and np.isfinite(losses).all()
                assert abs(float(losses.mean())-complete['ce'])<1e-8
                assert abs(float(losses[1024:].mean())-complete['post_eviction_ce'])<1e-8
                row={'predictions':len(losses),'elapsed_seconds':complete['elapsed_seconds']}
                for region,part in [('first1024',losses[:1024]),('post_eviction',losses[1024:]),('beyond10windows',losses[10240:]),('last64k',losses[-65536:])]:
                    ce=float(part.mean());row[region]={'ce':ce,'ppl':math.exp(min(ce,700))}
                result[group][label][policy]=row
    result['paired_trained_vs_standard']={}
    for arm in ['legacy','calibrated']:
        result['paired_trained_vs_standard'][arm]={}
        for policy in ['fifo','streamingllm']:
            paths=[root/'results'/(label+'-'+policy) for label in ['standard',arm]]
            if not all((p/'complete.json').exists() for p in paths):continue
            identities=[json.loads((p/'identity.json').read_text()) for p in paths]
            assert identities[0]['test_stream_sha256']==identities[1]['test_stream_sha256']
            boundaries=[json.loads((p/'complete.json').read_text())['boundaries'] for p in paths]
            assert boundaries[0]==boundaries[1]
            baseline,variant=[np.load(p/'losses.npy') for p in paths]
            delta=float((variant[1024:]-baseline[1024:]).mean())
            result['paired_trained_vs_standard'][arm][policy]={'post_eviction_ce_delta':delta,'ppl_change_percent':100*math.expm1(delta)}
    (root/('INCOMPLETE.json' if result['missing'] else 'complete.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['missing']:raise SystemExit(2)


if __name__=='__main__':main()
