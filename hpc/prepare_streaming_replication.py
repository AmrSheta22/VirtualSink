"""CPU Slurm preparation: isolated legacy dependencies, pinned models and PG19 books."""
import hashlib,json,os,subprocess,urllib.request
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,snapshot_download
from tokenizers import Tokenizer
from legacy_streaming_dependencies import install

ROOT=Path(os.environ['REPLICATION_ROOT'])


def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
    return path.read_bytes()


def main():
    assert os.environ.get('SLURM_JOB_ID')
    install(ROOT)
    upstream=json.load(urllib.request.urlopen('https://api.github.com/repos/mit-han-lab/streaming-llm/commits/main'))['sha']
    folder=ROOT/'upstream';folder.mkdir()
    for name in ['streaming_llm/kv_cache.py','streaming_llm/pos_shift/modify_gpt_neox.py','examples/eval_long_ppl.py']:
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
        fetch('https://raw.githubusercontent.com/mit-han-lab/streaming-llm/'+upstream+'/'+name,p)
    (folder/'streaming_llm/pos_shift/__init__.py').write_text('')
    (folder/'streaming_llm/__init__.py').write_text('')
    api=HfApi();dataset_revision=api.dataset_info('deepmind/pg19').sha
    data=ROOT/'data';data.mkdir()
    split_names={}
    for split in ['train','validation','test']:
        raw=fetch(f'https://huggingface.co/datasets/deepmind/pg19/resolve/{dataset_revision}/data/{split}_files.txt',data/(split+'-files.txt'))
        split_names[split]=sorted(raw.decode().splitlines())
    assert not(set(split_names['train'])&set(split_names['test']))
    assert not(set(split_names['validation'])&set(split_names['test']))
    models={}
    for label,repo in [('pythia28','EleutherAI/pythia-2.8b'),('pythia410','EleutherAI/pythia-410m')]:
        revision=api.model_info(repo).sha
        path=snapshot_download(repo,revision=revision,local_dir=str(ROOT/'models'/label),
             allow_patterns=['*.json','*.safetensors','*.bin','*.model','*.txt'])
        models[label]={'repo':repo,'revision':revision,'path':path}
    qpath=Path('/cluster/users/alex116u1/data/qwen15-matched-seeds-20261010/data/manifest.json')
    q=json.loads(qpath.read_text());models['qwen15']={'repo':q['model'],'revision':q['revision'],'path':q['model_path']}
    identities={};tokenizers={label:Tokenizer.from_file(str(Path(v['path'])/'tokenizer.json')) for label,v in models.items()}
    # Preserve separate book tokenization, order, and no scored cross-book boundary target.
    test_tokens={label:[] for label in models};counts={label:0 for label in models}
    identities['test']=[]
    for name in split_names['test']:
        text=fetch('https://storage.googleapis.com/deepmind-gutenberg/'+name,data/Path(name).name).decode('utf-8')
        identities['test'].append({'file':name,'sha256':hashlib.sha256(text.encode()).hexdigest()})
        for label,tok in tokenizers.items():
            ids=np.asarray(tok.encode(text).ids,dtype='<i8');test_tokens[label].append(ids);counts[label]+=len(ids)-1
        if min(counts.values())>=400000:break
    for label,books in test_tokens.items():
        np.savez(data/(label+'-test.npz'),**{'book%03d'%i:ids for i,ids in enumerate(books)})
    tok=tokenizers['pythia410']
    for split,budget in [('train',1050624),('validation',32768)]:
        ids=[];identity=[]
        for name in split_names[split]:
            p=data/(split+'-'+Path(name).name)
            text=fetch('https://storage.googleapis.com/deepmind-gutenberg/'+name,p).decode('utf-8')
            identity.append({'file':name,'sha256':hashlib.sha256(text.encode()).hexdigest()})
            ids.extend(tok.encode(text).ids);ids.append(tok.token_to_id('<|endoftext|>'))
            if len(ids)>=budget:break
        assert len(ids)>=budget
        np.asarray(ids[:budget],dtype='<i8').tofile(data/('pythia410-'+split+'.bin'));identities[split]=identity
    manifest={'models':models,'upstream_commit':upstream,'dataset_revision':dataset_revision,'books':identities,
              'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*') if p.is_file()}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':main()
