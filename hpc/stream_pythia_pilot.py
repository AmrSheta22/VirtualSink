"""Evaluate matched final pilot checkpoints after replay and decoder parity checks."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_pythia_init_pilot import adapt,validation
from run_streaming_replication import atomic
from virtualsink.uptraining.neox_streaming import NeoXDecoder


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--arm',choices=['standard','legacy','calibrated'],required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    root=a.root;arm=a.arm;pilot=root/'pilot'/arm
    manifest=json.loads((root/'manifest.json').read_text())
    # The saved Pythia2.8B stream is reusable only if tokenization is identical.
    tokenizers=[AutoTokenizer.from_pretrained(manifest['models'][k]['path']) for k in ['pythia410','pythia28']]
    strings=[t.backend_tokenizer.to_str() for t in tokenizers]
    assert strings[0]==strings[1], 'tokenizers differ; prepare a separate test stream'
    tokenizer_sha=hashlib.sha256(strings[0].encode()).hexdigest()
    summary=json.loads((pilot/'summary.json').read_text())
    reference=json.loads((a.reference_root/'pilot'/arm/'summary.json').read_text())
    metrics=[json.loads(x) for x in (pilot/'metrics.jsonl').read_text().splitlines()]
    reference_metrics=[json.loads(x) for x in (a.reference_root/'pilot'/arm/'metrics.jsonl').read_text().splitlines()]
    assert summary['steps']==256 and metrics[-1]['step']==256
    # Verify every validation checkpoint, not just the final result.
    actual=[summary['initial']['validation_ce']]+[r['validation_ce'] for r in metrics if 'validation_ce' in r]
    expected=[reference['initial']['validation_ce']]+[r['validation_ce'] for r in reference_metrics if 'validation_ce' in r]
    replay_error=max(abs(x-y) for x,y in zip(actual,expected))
    assert len(actual)==len(expected)==9 and replay_error<.001, (arm,replay_error)
    model=AutoModelForCausalLM.from_pretrained(manifest['models']['pythia410']['path'],torch_dtype=torch.float32)
    adapt(model,arm)
    checkpoint=pilot/'final-updates.pt';updates=torch.load(checkpoint,map_location='cpu')
    assert set(updates)=={n for n,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(updates,strict=False);del updates
    model.cuda().eval()
    val=np.memmap(root/'data/pythia410-validation.bin',dtype='<i8',mode='r')
    restored_ce=validation(model,val)
    assert abs(restored_ce-metrics[-1]['validation_ce'])<.0001
    with np.load(root/'data/pythia28-test.npz') as data:
        tokens=torch.tensor(data['book000'][:1153],device='cuda')
    # Parallel, cached absolute FIFO, and official raw-key rebasing oracles.
    parallel=model(tokens[:64][None],use_cache=False).logits[0]
    decoder=NeoXDecoder(model,1024)
    for i,t in enumerate(tokens[:64]):torch.testing.assert_close(decoder.step(t),parallel[i],atol=.003,rtol=.0001)
    del decoder,parallel
    from streaming_llm.kv_cache import StartRecentKVCache
    from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention
    parity={}
    for policy in ['fifo','streamingllm']:
        rebase=policy=='streamingllm';prefix=4 if rebase else 0
        if rebase:enable_gpt_neox_pos_shift_attention(model)
        decoder=NeoXDecoder(model,1024,prefix,rebase);cache=StartRecentKVCache(prefix,1024-prefix);past=None;maximum=0.
        for i,t in enumerate(tokens):
            position=len(past[0][0][0,0]) if rebase and past is not None else (0 if rebase else i)
            output=model(t.reshape(1,1),past_key_values=past,position_ids=torch.tensor([[position]],device='cuda'),use_cache=True)
            past=cache(output.past_key_values);actual_logits=decoder.step(t)
            maximum=max(maximum,(actual_logits-output.logits[0,0]).abs().max().item())
            torch.testing.assert_close(actual_logits,output.logits[0,0],atol=.003,rtol=.0001)
        parity[policy]={'tokens':len(tokens),'maximum_absolute_logit_error':maximum}
        del decoder,past,output
        print('PARITY_PASSED',arm,policy,maximum,flush=True)
    identity={'arm':arm,'checkpoint_step':256,'checkpoint_sha256':sha(checkpoint),
              'model':manifest['models']['pythia410'],'test_tokenizer_sha256':tokenizer_sha,
              'test_stream_sha256':sha(root/'data/pythia28-test.npz'),'restored_validation_ce':restored_ce,
              'replay_max_validation_ce_error':replay_error,'parity':parity,
              'capacity':1024,'transient_attention_capacity':1025,'dtype':'float32',
              'book_boundary':'separate-book targets, persistent cache, no scored cross-book target'}
    atomic(pilot/'streaming-validation.json',identity)
    for policy in ['fifo','streamingllm']:
        folder=root/'results'/(arm+'-'+policy);folder.mkdir(parents=True,exist_ok=False)
        atomic(folder/'identity.json',dict(identity,policy=policy,budget=400000))
        decoder=NeoXDecoder(model,1024,4 if policy=='streamingllm' else 0,policy=='streamingllm')
        started=time.monotonic();losses=[];boundaries=[]
        with np.load(root/'data/pythia28-test.npz') as data:
            for book in sorted(data.files):
                ids=torch.tensor(data[book],device='cuda');boundaries.append({'book':book,'prediction_start':len(losses)})
                for i in range(len(ids)-1):
                    logits=decoder.step(ids[i]);loss=torch.nn.functional.cross_entropy(logits[None],ids[i+1:i+2]).item()
                    assert math.isfinite(loss);losses.append(loss)
                    if len(losses)%512==0:
                        atomic(folder/'progress.json',{'predictions':len(losses),'elapsed_seconds':time.monotonic()-started,
                               'seconds_per_prediction':(time.monotonic()-started)/len(losses),'recent_ce':sum(losses[-512:])/512})
                        print('STREAM',arm,policy,len(losses),flush=True)
                    if len(losses)==400000:break
                if len(losses)==400000:break
        assert len(losses)==400000
        np.save(folder/'losses.npy',np.asarray(losses,dtype='<f8'))
        atomic(folder/'complete.json',{'predictions':len(losses),'ce':sum(losses)/len(losses),
               'post_eviction_ce':sum(losses[1024:])/(len(losses)-1024),'elapsed_seconds':time.monotonic()-started,'boundaries':boundaries})
        del decoder


if __name__=='__main__':main()
