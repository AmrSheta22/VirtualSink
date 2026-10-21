"""Reconstruct WikiText articles from authoritative top-level title rows."""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from virtualsink.uptraining.data import sha256


def articles(rows):
    text, start = [], None
    for index, row in enumerate(rows):
        if re.match(r'^=\s+[^=].*?\s+=$', row.strip()):
            if start is not None:
                yield start, index, '\n'.join(text)
            text, start = [row], index
        elif start is not None:
            text.append(row)
    # A shard can end inside an article. Discard its last unclosed article.


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(os.environ['QWEN_LONG_ROOT'])
    previous=json.loads((Path.home()/'data/phase3-k80-20260909/data/qwen05/manifest.json').read_text())
    if previous['model']!='Qwen/Qwen2.5-0.5B' or previous['revision']!='060db6499f32faf8b98477b0a26969ef7d8b9987':
        raise ValueError('unexpected pretrained identity')
    model=Path(previous['model_path'])
    tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
    config=json.loads((model/'config.json').read_text())
    eos=config['eos_token_id']
    corpus=Path.home()/'data/phase3-k80-20260909/hf-cache/hub/datasets--Salesforce--wikitext/snapshots'/previous['source']['revision']
    destination=root/'data';destination.mkdir(exist_ok=False)
    manifest={'format':'document-uint32-v1','model':previous['model'],'revision':previous['revision'],
        'model_path':str(model),'eos_token_id':eos,'source':previous['source'],
        'packing':'complete articles bounded by top-level title rows; discard shard-edge partial articles; no cross-document attention',
        'article_token_limits':[1025,16384], 'splits':{},
        'pretrained_sha256':{p.name:sha256(p) for p in model.iterdir() if p.suffix=='.safetensors' or p.name in ('config.json','tokenizer.json')}}
    seen=set()
    for split in ('train','validation','test'):
        offsets=[0];records=[];outpath=destination/(split+'.bin')
        with outpath.open('wb') as output:
            for source in previous['splits'][split]['source_files']:
                path=corpus/source['file']
                if sha256(path)!=source['sha256']:raise ValueError('raw source changed')
                rows=pq.read_table(path,columns=['text'])['text'].to_pylist()
                for start,end,text in articles(rows):
                    ids=tokenizer.encode(text,add_special_tokens=False).ids+[eos]
                    if not 1025<=len(ids)<=16384:continue
                    array=np.asarray(ids,dtype='<u4');digest=hashlib.sha256(array.tobytes()).hexdigest()
                    if digest in seen:continue
                    seen.add(digest);array.tofile(output);offsets.append(offsets[-1]+len(ids))
                    records.append({'file':source['file'],'first_row':start,'exclusive_end_row':end,'tokens':len(ids),'sha256':digest})
                    if split=='train' and offsets[-1]>=2_000_000:break
                if split=='train' and offsets[-1]>=2_000_000:break
        np.save(destination/(split+'.offsets.npy'),np.asarray(offsets,dtype=np.int64),allow_pickle=False)
        if split=='train' and offsets[-1]<1_100_000:raise ValueError('insufficient nonrepeating training corpus')
        if split!='train' and sum(r['tokens']>4096 for r in records)<4:raise ValueError('need four long held-out articles')
        manifest['splits'][split]={'file':outpath.name,'offsets_file':split+'.offsets.npy','tokens':offsets[-1],
            'documents':len(records),'sha256':sha256(outpath),'offsets_sha256':sha256(destination/(split+'.offsets.npy')),
            'source_files':previous['splits'][split]['source_files'],'articles':records}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'PREPARATION_PASSED','splits':{s:{k:v[k] for k in ['tokens','documents']} for s,v in manifest['splits'].items()}}),flush=True)


if __name__=='__main__':main()
