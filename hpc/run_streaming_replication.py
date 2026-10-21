"""PG19 pretrained replication screen: bounded KV, official oracle, and recomputation."""
import argparse,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from virtualsink.uptraining.neox_streaming import NeoXDecoder
from virtualsink.uptraining.native import load_native,set_window
from virtualsink.uptraining.qwen import disable_sinks
from virtualsink.uptraining.streaming_fifo import StreamingDecoder


class QwenDecoder:
    def __init__(self,model,capacity,prefix,rebase):
        self.inner=StreamingDecoder(model,capacity+1,prefix,rebase)
        self.capacity,self.prefix=capacity,prefix
    def step(self,token):
        logits,*_=self.inner.step(token)
        for i,(k,v) in enumerate(self.inner.cache):
            if k.shape[2]>self.capacity:
                n=self.capacity-self.prefix
                self.inner.cache[i]=(torch.cat((k[:,:,:self.prefix],k[:,:,-n:]),2),torch.cat((v[:,:,:self.prefix],v[:,:,-n:]),2))
        return logits


def load(root,label):
    info=json.loads((root/'manifest.json').read_text())['models'][label]
    if label=='qwen15':
        model=load_native(info['path'],32).cuda().eval();disable_sinks(model);return model
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(info['path'],torch_dtype=torch.float16).cuda().eval()


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


@torch.no_grad()
def validate(root):
    from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
    from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention
    from streaming_llm.kv_cache import StartRecentKVCache
    torch.manual_seed(83)
    cfg=GPTNeoXConfig(vocab_size=61,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
                     intermediate_size=64,max_position_embeddings=128,rotary_pct=.5,
                     attention_dropout=0,hidden_dropout=0)
    tiny=GPTNeoXForCausalLM(cfg).cuda().eval();ids=torch.randint(0,61,(97,),device='cuda')
    # Absolute cached decoder agrees with parallel HF before any eviction.
    parallel=tiny(ids[None]).logits[0];decoder=NeoXDecoder(tiny,128)
    for i,t in enumerate(ids):torch.testing.assert_close(decoder.step(t),parallel[i],atol=1e-5,rtol=1e-4)
    enable_gpt_neox_pos_shift_attention(tiny)
    for prefix in [0,4]:
        decoder=NeoXDecoder(tiny,32,prefix,True);cache=StartRecentKVCache(prefix,32-prefix);past=None
        for token in ids:
            ref=tiny(token.reshape(1,1),past_key_values=past,use_cache=True)
            past=cache(ref.past_key_values)
            torch.testing.assert_close(decoder.step(token),ref.logits[0,0],atol=1e-5,rtol=1e-4)
    del tiny;torch.cuda.empty_cache()
    # Validate the pilot attention replacement against untouched HF and exercise a real update.
    from train_pythia_init_pilot import adapt,hidden_loss
    tiny=GPTNeoXForCausalLM(cfg).cuda().eval()
    expected=tiny(ids[:64][None],use_cache=False).logits.detach()
    # adapt creates CPU sink parameters; move them with the model afterwards.
    adapt(tiny,'standard');tiny.cuda()
    torch.testing.assert_close(tiny(ids[:64][None],use_cache=False).logits,expected,atol=1e-5,rtol=1e-4)
    for l in tiny.gpt_neox.layers:
        l.attention.sink_disabled=False;l.attention.sink_b.fill_(-20.)
    # Exit no_grad for a controlled language-loss update; values are fixed zero in this pilot.
    with torch.enable_grad():
        for l in tiny.gpt_neox.layers:
            l.attention.sink_w.requires_grad_(True);l.attention.sink_b.requires_grad_(True)
        tiny.train();tiny.gradient_checkpointing_enable();tiny.enable_input_require_grads()
        loss=hidden_loss(tiny,ids[:64][None]);loss.backward()
        assert all(l.attention.sink_b.grad is not None and torch.isfinite(l.attention.sink_b.grad).all() for l in tiny.gpt_neox.layers)
    del tiny,expected;torch.cuda.empty_cache()
    print('TINY_PARITY_AND_PILOT_GRADIENTS_PASSED',flush=True)
    results={}
    for label in ['pythia28','qwen15']:
        model=load(root,label)
        with np.load(root/'data'/(label+'-test.npz')) as data:tokens=torch.tensor(data['book000'][:1153],device='cuda')
        if label=='pythia28':
            decoder=NeoXDecoder(model,2048);past=None;maximum=0.
            for token in tokens[:129]:
                expected=model(token.reshape(1,1),past_key_values=past,use_cache=True);past=expected.past_key_values
                actual=decoder.step(token);maximum=max(maximum,(actual-expected.logits[0,0]).abs().max().item())
                torch.testing.assert_close(actual,expected.logits[0,0],atol=.03,rtol=.002)
            del past,expected,decoder
            print('FULL_MODEL_PRE_EVICTION_PARITY_PASSED',label,maximum,flush=True)
            enable_gpt_neox_pos_shift_attention(model)
            actual=NeoXDecoder(model,1024,4,True);cache=StartRecentKVCache(4,1020);past=None
            for token in tokens:
                expected=model(token.reshape(1,1),past_key_values=past,use_cache=True);past=cache(expected.past_key_values)
                logits=actual.step(token)
                torch.testing.assert_close(logits,expected.logits[0,0],atol=.03,rtol=.002)
            results[label]={'pre_eviction_max_logit_error':maximum,'official_post_eviction_parity_tokens':len(tokens)}
            del past,expected,actual
            print('OFFICIAL_POST_EVICTION_PARITY_PASSED',label,flush=True)
        else:
            from virtualsink.uptraining.evaluate_streaming import check_parity
            from virtualsink.uptraining.qwen15_streaming import rebase_parity
            results[label]={'parallel':check_parity(model,tokens,1025),'rebase_oracle':rebase_parity(model,tokens)}
            from virtualsink.uptraining.rebased_reference import RebasedReference
            production=QwenDecoder(model,1024,4,True);oracle=RebasedReference(model,1025,4)
            for token in tokens:
                torch.testing.assert_close(production.step(token),oracle.step(token),atol=.003,rtol=.0001)
            results[label]['official_post_trim_convention_oracle_tokens']=len(tokens)
            del production,oracle
        del model;torch.cuda.empty_cache()
    atomic(root/'validation.json',results);print('VALIDATION_PASSED',flush=True)


@torch.no_grad()
def evaluate(root,label,policy,budget):
    model=load(root,label);capacity=1024
    folder=root/'results'/(label+'-'+policy);folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    atomic(folder/'identity.json',{'model':manifest['models'][label],'policy':policy,'capacity':capacity,
           'budget':budget,'transient_attention_capacity':capacity+1,'book_boundary':'official separate-book predictions; persistent cache; no scored cross-book boundary'})
    prefix=4 if policy=='streamingllm' else 0;rebase=policy=='streamingllm'
    decoder=(NeoXDecoder(model,capacity,prefix,rebase) if label=='pythia28' else QwenDecoder(model,capacity,prefix,rebase))
    started=time.monotonic();losses=[];history=[];boundaries=[]
    with np.load(root/'data'/(label+'-test.npz')) as data:
        for book in sorted(data.files):
            ids=torch.tensor(data[book],device='cuda');boundaries.append({'book':book,'prediction_start':len(losses)})
            for i in range(len(ids)-1):
                if policy=='recompute':
                    history.append(int(ids[i]));history=history[-capacity-1:]
                    inputs=torch.tensor(history,device='cuda')[None]
                    if label=='pythia28':hidden=model.gpt_neox(inputs,use_cache=False).last_hidden_state
                    else:hidden=model.model(inputs).last_hidden_state
                    logits=(model.embed_out(hidden[:,-1]) if label=='pythia28' else model.lm_head(hidden[:,-1]))[0]
                else:logits=decoder.step(ids[i])
                loss=torch.nn.functional.cross_entropy(logits.float()[None],ids[i+1:i+2]).item()
                if not math.isfinite(loss):raise FloatingPointError('nonfinite prediction')
                losses.append(loss)
                if len(losses)%512==0:
                    atomic(folder/'progress.json',{'predictions':len(losses),'elapsed_seconds':time.monotonic()-started,
                         'seconds_per_prediction':(time.monotonic()-started)/len(losses),'recent_ce':sum(losses[-512:])/512})
                    print('PREDICTIONS',len(losses),flush=True)
                if len(losses)>=budget:break
            if len(losses)>=budget:break
    assert len(losses)==budget
    np.save(folder/'losses.npy',np.asarray(losses,dtype='<f8'))
    atomic(folder/'complete.json',{'predictions':len(losses),'ce':sum(losses)/len(losses),
           'post_eviction_ce':sum(losses[1024:])/(len(losses)-1024),'elapsed_seconds':time.monotonic()-started,'boundaries':boundaries})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--validate',action='store_true')
    p.add_argument('--label',choices=['pythia28','qwen15']);p.add_argument('--policy',choices=['fifo','streamingllm','recompute'])
    p.add_argument('--budget',type=int,default=400000);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    if a.validate:validate(a.root)
    else:evaluate(a.root,a.label,a.policy,a.budget)


if __name__=='__main__':main()
