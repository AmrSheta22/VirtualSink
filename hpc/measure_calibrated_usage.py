"""Read-only forward diagnostics of calibrated zero-value sink attention mass."""
import argparse,json,types,hashlib,time,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from train_pythia_init_pilot import adapt
from virtualsink.uptraining.neox_streaming import NeoXDecoder


def summary(values):
    # Values are [layer, head, query]; all statistics weight queries/heads equally.
    v=values.astype(np.float64);means=v.mean(-1)
    top=np.unravel_index(np.argsort(means.ravel())[-10:][::-1],means.shape)
    return {'queries':v.shape[-1],'layers':v.shape[0],'heads_per_layer':v.shape[1],
            'mean_sink_mass':float(v.mean()),'median_sink_mass':float(np.median(v)),
            'p90_sink_mass':float(np.quantile(v,.9)),'p99_sink_mass':float(np.quantile(v,.99)),
            'maximum_sink_mass':float(v.max()),
            'fraction_head_queries_above':{str(t):float((v>t).mean()) for t in [1e-6,1e-5,1e-4,.001,.01,.1,.5]},
            'mean_by_layer':means.mean(1).tolist(),'mean_by_head':means.tolist(),
            'heads_mean_above':{str(t):int((means>t).sum()) for t in [1e-5,.001,.01,.1]},
            'top_heads':[{'layer':int(l),'head':int(h),'mean_sink_mass':float(means[l,h])} for l,h in zip(*top)]}


class Observer:
    def __init__(self,model):
        self.values=[[] for _ in model.gpt_neox.layers];self.enabled=False;self.packed=False
        for index,layer in enumerate(model.gpt_neox.layers):
            original=layer.attention._attn
            def observed(att,q,k,v,attention_mask=None,head_mask=None,_index=index,_original=original):
                output=_original(q,k,v,attention_mask,head_mask)
                if self.enabled:
                    n,m=q.shape[-2],k.shape[-2]
                    scores=(q.float()@k.float().transpose(-1,-2))*att.norm_factor
                    pos=torch.arange(m,device=q.device);scores=scores.masked_fill(~(pos[None,:]<=pos[m-n:,None]),-torch.inf)
                    if attention_mask is not None:scores=scores+attention_mask
                    logits=(q.float()*att.sink_w[None,:,None]).sum(-1)+att.sink_b[None,:,None]
                    # Compute p_sink directly; 1-gate loses tiny masses to FP32 cancellation.
                    mass=torch.sigmoid(logits-scores.logsumexp(-1))[0]
                    if self.packed:mass=mass[:,:-1]
                    self.values[_index].append(mass.cpu().numpy())
                return output
            layer.attention._attn=types.MethodType(observed,layer.attention)
    def reset(self,packed=False):
        self.values=[[] for _ in self.values];self.packed=packed;self.enabled=True
    def array(self):return np.stack([np.concatenate(parts,axis=-1) for parts in self.values])


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=json.loads((a.root/'manifest.json').read_text());pilot=a.root/'pilot/calibrated'
    initial=json.loads((pilot/'initial.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(manifest['models']['pythia410']['path'],torch_dtype=torch.float32)
    adapt(model,'calibrated')
    for i,l in enumerate(model.gpt_neox.layers):l.attention.sink_b.copy_(torch.tensor(initial['biases'][i]))
    model.cuda().eval();observer=Observer(model)
    val=np.memmap(a.root/'data/pythia410-validation.bin',dtype='<i8',mode='r')
    result={'scope':'Frozen calibrated final256 checkpoint. Packed32x511 prediction queries. Streaming first4096 PG19 predictions per policy, one book, cache1024/transient1025. Not full400k usage.',
            'mass_definition':'p_sink = sigmoid(query_conditioned_sink_logit - logsumexp(real_attention_logits)); zero-value sinks attenuate real attention output by 1-p_sink.'}
    for stage in ['initial','trained']:
        if stage=='trained':
            updates=torch.load(pilot/'final-updates.pt',map_location='cpu');model.load_state_dict(updates,strict=False);del updates
        observer.reset(packed=True)
        for i in range(32):model.gpt_neox(torch.tensor(val[i*512:(i+1)*512].copy(),device='cuda')[None],use_cache=False)
        values=observer.array();np.save(a.out/(stage+'-packed-mass.npy'),values);result[stage+'_packed']=summary(values)
        observer.enabled=False
        (a.out/'progress.json').write_text(json.dumps(result,indent=2));print('PACKED_DONE',stage,result[stage+'_packed']['mean_sink_mass'],flush=True)
    weights=np.stack([l.attention.sink_w.cpu().numpy() for l in model.gpt_neox.layers]);biases=np.stack([l.attention.sink_b.cpu().numpy() for l in model.gpt_neox.layers]);norms=np.linalg.norm(weights,axis=-1)
    result['parameters']={'score_weight_norm_mean':float(norms.mean()),'score_weight_norm_max':float(norms.max()),'score_weight_nonzero_heads':int((norms>0).sum()),'bias_delta_max_absolute':float(np.abs(biases-np.asarray(initial['biases'])).max())}
    with np.load(a.root/'data/pythia28-test.npz') as data:tokens=torch.tensor(data['book000'][:4097],device='cuda')
    # Instrumentation must not alter model outputs.
    observer.enabled=False;expected=model(tokens[:64][None],use_cache=False).logits
    observer.reset();actual=model(tokens[:64][None],use_cache=False).logits
    torch.testing.assert_close(actual,expected,atol=0,rtol=0);result['instrumentation_logit_error']=0;del actual,expected
    for policy in ['fifo','streamingllm']:
        observer.reset();decoder=NeoXDecoder(model,1024,4 if policy=='streamingllm' else 0,policy=='streamingllm');begin=time.monotonic()
        for i,token in enumerate(tokens[:-1]):
            decoder.step(token)
            if (i+1)%512==0:print('STREAM',policy,i+1,'elapsed',time.monotonic()-begin,flush=True)
        values=observer.array();np.save(a.out/(policy+'-mass.npy'),values)
        result[policy]={'all':summary(values),'pre_eviction':summary(values[:,:,:1024]),'post_eviction':summary(values[:,:,1024:]),'last1024':summary(values[:,:,-1024:]),'elapsed_seconds':time.monotonic()-begin}
        observer.enabled=False;del decoder
        (a.out/'progress.json').write_text(json.dumps(result,indent=2));print('STREAM_DONE',policy,flush=True)
    result['checkpoint_identity']=json.loads((pilot/'streaming-validation.json').read_text())
    result['elapsed_seconds']=time.monotonic()-started
    (a.out/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)


if __name__=='__main__':main()
