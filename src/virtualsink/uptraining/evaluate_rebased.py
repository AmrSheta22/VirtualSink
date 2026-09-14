"""One-article SmolLM prefix-cache positional-rebasing ablation."""
import json,os,re,functools,math
from pathlib import Path
import torch
import numpy as np
from .native import load_native
from .qwen import disable_sinks
from .data import sha256
from . import evaluate_streaming as ev
from .streaming_fifo import StreamingDecoder

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    base=Path.home()/'data'
    old=base/'uptrained-streaming-20260913/results/smollm135'
    identity=json.loads((old/'config.json').read_text())
    config=identity['training_config']['standard']; manifest=config['manifest']
    checkpoint=base/'phase3-swa-20260910/runs/smollm135-standard/best.pt'
    assert sha256(checkpoint)==identity['checkpoint_sha256']['standard']
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    source=next((base/'phase3-k80-20260909/hf-cache/hub').glob('datasets--Salesforce--wikitext/snapshots/'+manifest['source']['revision']+'/wikitext-103-raw-v1/test-*.parquet'))
    assert sha256(source)==manifest['splits']['test']['source_files'][0]['sha256']
    parts=[]; active=False; title=identity['articles'][0]['title']
    for row in pq.read_table(source,columns=['text']).column('text').to_pylist():
        if re.match(r'^\s*= [^=].*[^=] =\s*$',row):
            if active:break
            active=row.strip()==title
        if active:parts.append(row)
    import hashlib
    text=''.join(parts)
    assert hashlib.sha256(text.encode()).hexdigest()==identity['articles'][0]['text_sha256']
    tok=Tokenizer.from_file(str(Path(manifest['model_path'])/'tokenizer.json'))
    tokens=torch.tensor(tok.encode(text).ids[:8193],device='cuda')
    out=base/'smollm-rebased-20260914';out.mkdir(exist_ok=False)
    ev.write(out/'config.json',{'article':identity['articles'][0],'checkpoint_sha256':sha256(checkpoint),
      'position_policy':'unrotated cached keys, cache-relative RoPE indices 0..511, query at final cache index',
      'capacity':512,'prefix':4,'predictions':8192,'job':os.environ['SLURM_JOB_ID']})
    m=load_native(manifest['model_path'],64).eval();saved=torch.load(checkpoint,map_location='cpu')
    m.load_state_dict(saved['model']);del saved;m.cuda();disable_sinks(m)
    a=StreamingDecoder(m,512,4);b=StreamingDecoder(m,512,4,rebase=True)
    for t in tokens[:512]:
        x,*_=a.step(t);y,*_=b.step(t);torch.testing.assert_close(x,y,atol=.003,rtol=.0001)
    ev.write(out/'parity.json',{'before_eviction_tokens':512,'passed':True})
    ev.StreamingDecoder=functools.partial(StreamingDecoder,rebase=True)
    ev.evaluate(m,tokens,tok,4,out,'standard-rebased-prefix4')
    results={}
    for name in ['standard-fifo','standard-prefix4','query_sink-fifo']:
        results[name]=json.loads((old/f'{name}-article0.json').read_text())
    results['standard-rebased-prefix4']=json.loads((out/'standard-rebased-prefix4.json').read_text())
    summary={name:{region:math.exp(float(np.mean(r['per_token_ce'][a:b]))) for region,a,b in
      [('before_eviction',0,512),('post_eviction',512,8192),('beyond10w',5120,8192)]} for name,r in results.items()}
    ev.write(out/'comparison.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
