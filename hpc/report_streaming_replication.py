"""Require complete screens; score identical first20k target ranges per model."""
import json,os,math
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(os.environ['REPLICATION_ROOT']);rows={};missing=[]
    for label in ['pythia28','qwen15']:
        rows[label]={}
        for policy in ['fifo','streamingllm','recompute']:
            folder=root/'results'/(label+'-'+policy)
            if not (folder/'complete.json').exists():missing.append(str(folder));continue
            v=np.load(folder/'losses.npy');assert np.isfinite(v).all()
            expected=20000 if policy=='recompute' else 400000
            assert v.shape==(expected,)
            rows[label][policy]={'predictions':len(v),'first20k_ce':float(v[:20000].mean()),
                'first20k_post_eviction_ce':float(v[1024:20000].mean()),'full_post_eviction_ce':float(v[1024:].mean())}
            for k,x in list(rows[label][policy].items()):
                if k.endswith('_ce'):rows[label][policy][k[:-3]+'_ppl']=math.exp(min(x,700))
    pilots={}
    for arm in ['standard','legacy','calibrated']:
        p=root/'pilot'/arm/'summary.json'
        if p.exists():pilots[arm]=json.loads(p.read_text())
        else:missing.append(str(p))
    result={'metrics_by_model':rows,'initialization_pilot':pilots,'missing':missing,
            'scope':'400k PG19 screen; recomputation20k. Separate models have different tokenizations. Not full4million-token replication; pilot256 steps, one seed, not definitive.'}
    (root/('INCOMPLETE.json' if missing else 'complete.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if missing:raise SystemExit(2)


if __name__=='__main__':main()
