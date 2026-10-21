"""Train and evaluate four genuine prefix-like KV sinks on frozen Pythia410M."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from virtualsink.uptraining.prefix_kv import capture,install,PrefixKVDecoder
from virtualsink.uptraining.neox_streaming import NeoXDecoder

ASSETS=Path('/cluster/users/alex116u1/data/streamingllm-replication-20261013-r3')
OLD=Path('/cluster/users/alex116u1/data/pythia-positive-control-20261017')
POLICIES=['fifo','natural','fixed','frozen','trained','zero_value']
CAPACITY=1024
BUDGET=400000


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def load():
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    return AutoModelForCausalLM.from_pretrained(manifest['models']['pythia410']['path'],torch_dtype=torch.float32,local_files_only=True).cuda().eval()


def memory_state(model):
    return {n:p.detach().cpu() for n,p in model.named_parameters() if n.endswith(('.virtual_keys','.virtual_values'))}


def add_memory(model,root,trained=False,zero=False):
    initial=torch.load(root/'initial.pt',map_location='cpu')
    pairs=[(p['key'].cuda(),p['value'].cuda()) for p in initial]
    install(model,pairs,CAPACITY)
    if trained:
        updates=torch.load(root/'final.pt',map_location='cpu')
        assert set(updates)==set(memory_state(model));model.load_state_dict(updates,strict=False)
    if zero:
        with torch.no_grad():
            for l in model.gpt_neox.layers:l.attention.virtual_values.zero_()
    return model


@torch.no_grad()
def oracle_parity(root,trained=False,length=1153):
    """Independent HF official forward seeded with the same raw prefix KV."""
    from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention
    from streaming_llm.kv_cache import StartRecentKVCache
    student=add_memory(load(),root,trained)
    teacher=load();enable_gpt_neox_pos_shift_attention(teacher)
    pairs=[(l.attention.virtual_keys[None].detach().clone(),l.attention.virtual_values[None].detach().clone()) for l in student.gpt_neox.layers]
    past=tuple(pairs);cache=StartRecentKVCache(4,1020)
    with np.load(ASSETS/'data/pythia28-test.npz') as data:ids=torch.tensor(data['book000'][:length],device='cuda')
    decoder=PrefixKVDecoder(student);maximum=0.
    for i,t in enumerate(ids):
        position=past[0][0].shape[2]
        reference=teacher(t.reshape(1,1),past_key_values=past,position_ids=torch.tensor([[position]],device='cuda'),use_cache=True)
        past=cache(reference.past_key_values);actual=decoder.step(t)
        maximum=max(maximum,(actual-reference.logits[0,0]).abs().max().item())
        torch.testing.assert_close(actual,reference.logits[0,0],atol=.003,rtol=.0001)
        assert all(k.shape[2]+4<=CAPACITY for k,v in decoder.cache)
        if (i+1)%256==0:print('ORACLE',trained,i+1,maximum,flush=True)
    del decoder,teacher,past,reference
    # Parallel eviction-aware training forward must match production streaming.
    decoder=PrefixKVDecoder(student);parallel=student(ids[None],use_cache=False).logits[0]
    parallel_error=0.;relative_error=0.;ce_error=0.
    for i,t in enumerate(ids):
        actual=decoder.step(t);parallel_error=max(parallel_error,(actual-parallel[i]).abs().max().item())
        relative_error=max(relative_error,((actual-parallel[i]).norm()/parallel[i].norm()).item())
        if i+1<len(ids):
            target=ids[i+1:i+2]
            ce_error=max(ce_error,abs(torch.nn.functional.cross_entropy(actual[None],target).item()-torch.nn.functional.cross_entropy(parallel[i:i+1],target).item()))
    # Parallel GEMMs and common-shift RoPE use a different FP32 operation order
    # than rolling single-query GEMMs. Gate the objective as well as logits;
    # the independent cached/HF comparison above retains its strict tolerance.
    print('PARALLEL_NUMERICS',parallel_error,relative_error,ce_error,flush=True)
    assert parallel_error<.01 and relative_error<.001 and ce_error<.001, (parallel_error,relative_error,ce_error)
    del student,decoder,parallel
    torch.cuda.empty_cache()
    return {'tokens':length,'oracle_max_logit_error':maximum,'parallel_streaming_max_logit_error':parallel_error,'parallel_streaming_max_relative_l2':relative_error,'parallel_streaming_max_target_ce_error':ce_error}


def prepare(root):
    torch.manual_seed(1729);model=load()
    train=np.memmap(ASSETS/'data/pythia410-train.bin',dtype='<i8',mode='r');prefix=torch.tensor(train[:4].copy(),device='cuda')
    pairs=capture(model,prefix)
    torch.save([{'key':k.cpu(),'value':v.cpu()} for k,v in pairs],root/'initial.pt')
    model_path=Path(json.loads((ASSETS/'manifest.json').read_text())['models']['pythia410']['path'])
    identity={'model':str(model_path),'anchor_token_ids':prefix.tolist(),'anchor_source':'first four PG19 training tokens; no test-derived initialization',
              'initial_sha256':sha(root/'initial.pt'),'assets':{str(p):sha(p) for p in [model_path/'model.safetensors',model_path/'config.json',ASSETS/'data/pythia28-test.npz',ASSETS/'data/pythia410-train.bin',ASSETS/'data/pythia410-validation.bin']},
              'slots':4,'cache_saved':1024,'cache_transient':1025,'parameters':sum(k.numel()+v.numel() for k,v in pairs),'precision':'float32','backbone':'frozen untouched pretrained Pythia410M'}
    write(root/'identity.json',identity)
    # Fixed real prompt and captured virtual slots must give the same logits.
    with torch.no_grad():
        tokens=torch.tensor(train[4:68].copy(),device='cuda')
        expected=model(torch.cat((prefix,tokens))[None],use_cache=False).logits[0,4:]
        install(model,pairs);actual=model(tokens[None],use_cache=False).logits[0]
        torch.testing.assert_close(actual,expected,atol=.003,rtol=.0001)
        fixed_error=(actual-expected).abs().max().item()
    # Exercise the actual checkpointed gradient path on tiny real batches.
    model.train();model.gradient_checkpointing_enable();model.enable_input_require_grads()
    loss=language_loss(model,tokens[None]);loss.backward()
    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
    # A saturated language batch may legitimately give an individual slot zero
    # gradient. Controlled per-slot connectivity is covered by test_prefix_kv.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for n,p in model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};optimizer.step()
    assert all(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad)
    del model,optimizer,before;torch.cuda.empty_cache()
    result=oracle_parity(root)
    result.update(fixed_prompt_max_logit_error=fixed_error,gradient_norms=grads)
    write(root/'validation.json',result);print('VALIDATION_PASSED',result['oracle_max_logit_error'],flush=True)


def language_loss(model,ids,post_only=False):
    hidden=model.gpt_neox(ids,use_cache=False).last_hidden_state[:,:-1]
    begin=1020 if post_only else 0;total=hidden.new_zeros(())
    for i in range(begin,hidden.shape[1],32):
        logits=model.embed_out(hidden[:,i:i+32]);targets=ids[:,i+1:i+33]
        total=total+torch.nn.functional.cross_entropy(logits.reshape(-1,model.config.vocab_size),targets.reshape(-1),reduction='sum')
    return total/(ids.shape[0]*(hidden.shape[1]-begin))


@torch.no_grad()
def validation(model,values):
    model.eval();return sum(language_loss(model,torch.tensor(values[i*2048:(i+1)*2048].copy(),device='cuda')[None],True).item() for i in range(16))/16


def train(root):
    torch.manual_seed(1729);model=add_memory(load(),root)
    train_data=np.memmap(ASSETS/'data/pythia410-train.bin',dtype='<i8',mode='r');val=np.memmap(ASSETS/'data/pythia410-validation.bin',dtype='<i8',mode='r')
    initial=validation(model,val);best=initial
    torch.save(memory_state(model),root/'best.pt');write(root/'training-initial.json',{'validation_post_eviction_ce':initial})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(s/25,1.) if s<25 else .5*(1+math.cos(math.pi*(s-25)/231)))
    model.gradient_checkpointing_enable();model.enable_input_require_grads();started=time.monotonic();best_step=0
    with (root/'metrics.jsonl').open('w') as log:
        for step in range(256):
            model.train();optimizer.zero_grad(set_to_none=True);total=0.
            for micro in range(2):
                begin=(step*2+micro)*2048;ids=torch.tensor(train_data[begin:begin+2048].copy(),device='cuda')[None]
                loss=language_loss(model,ids);assert torch.isfinite(loss);(loss/2).backward();total+=loss.item()/2
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            row={'step':step+1,'lm_ce':total,'elapsed_seconds':time.monotonic()-started,'lr':scheduler.get_last_lr()[0]}
            if (step+1)%32==0:
                ce=validation(model,val);row['validation_post_eviction_ce']=ce
                if ce<best:best=ce;best_step=step+1;torch.save(memory_state(model),root/'best.pt')
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    torch.save(memory_state(model),root/'final.pt')
    final=validation(model,val)
    write(root/'training-summary.json',{'steps':256,'input_tokens':1048576,'initial_validation_post_eviction_ce':initial,'final_validation_post_eviction_ce':final,'best_validation_post_eviction_ce':best,'best_step':best_step,'elapsed_seconds':time.monotonic()-started,'final_sha256':sha(root/'final.pt')})
    del model,optimizer;torch.cuda.empty_cache()
    write(root/'trained-validation.json',oracle_parity(root,True))


@torch.no_grad()
def evaluate(root,index):
    policy=POLICIES[index];model=load();identity=json.loads((root/'identity.json').read_text())
    for path,expected in identity['assets'].items():assert sha(path)==expected
    if policy in ['frozen','trained','zero_value']:
        add_memory(model,root,policy!='frozen',policy=='zero_value');decoder=PrefixKVDecoder(model)
    else:
        decoder=NeoXDecoder(model,CAPACITY,0 if policy=='fifo' else 4,True)
        if policy=='fixed':
            for token in torch.tensor(identity['anchor_token_ids'],device='cuda'):decoder.step(token)
    folder=root/'results'/policy;folder.mkdir(exist_ok=False)
    spec={'policy':policy,'identity_sha256':sha(root/'identity.json'),'predictions':BUDGET,'final_sha256':sha(root/'final.pt') if policy in ['trained','zero_value'] else None}
    write(folder/'identity.json',spec)
    started=time.monotonic();losses=[];boundaries=[];mass=None
    with np.load(ASSETS/'data/pythia28-test.npz') as data:
        for book in sorted(data.files):
            ids=torch.tensor(data[book],device='cuda');boundaries.append({'book':book,'prediction_start':len(losses)})
            for i in range(len(ids)-1):
                logits=decoder.step(ids[i]);ce=torch.nn.functional.cross_entropy(logits[None],ids[i+1:i+2]).item();assert math.isfinite(ce);losses.append(ce)
                if isinstance(decoder,PrefixKVDecoder):
                    observed=decoder.usage().double();mass=observed if mass is None else mass+observed
                if len(losses)%512==0:
                    row={'predictions':len(losses),'seconds_per_prediction':(time.monotonic()-started)/len(losses),'elapsed_seconds':time.monotonic()-started}
                    if mass is not None:row['mean_virtual_attention']=float((mass/len(losses)).sum(-1).mean())
                    write(folder/'progress.json',row);print(policy,json.dumps(row),flush=True)
                if len(losses)==BUDGET:break
            if len(losses)==BUDGET:break
    values=np.asarray(losses,dtype='<f8');np.save(folder/'losses.npy',values)
    if policy in ['fifo','natural']:
        label='pretrained-off-prefix'+('0' if policy=='fifo' else '4')+'-relative'
        old=np.load(OLD/'results'/label/'losses.npy');assert np.max(np.abs(values[:len(old)]-old))<.003
    write(folder/'complete.json',{'predictions':len(losses),'boundaries':boundaries,'ce':float(values.mean()),'post_eviction_ce':float(values[1024:].mean()),'elapsed_seconds':time.monotonic()-started,'mean_virtual_attention_by_head_slot':(mass/len(losses)).cpu().tolist() if mass is not None else None})


def report(root):
    result={'rows':{},'missing':[],'scope':'One-seed trained persistent four-KV prefix, frozen pretrained410M backbone;400k previously inspected PG19 predictions. All policies cache-relative,1024 total saved slots including virtual slots. Test-independent training-prefix initialization.'}
    for policy in POLICIES:
        folder=root/'results'/policy
        if not (folder/'complete.json').exists():result['missing'].append(policy);continue
        c=json.loads((folder/'complete.json').read_text());v=np.load(folder/'losses.npy');assert v.shape==(BUDGET,) and np.isfinite(v).all();assert abs(v.mean()-c['ce'])<1e-8
        row={'elapsed_seconds':c['elapsed_seconds']}
        for label,s in [('post_eviction',v[1024:]),('beyond10windows',v[10240:]),('last64k',v[-65536:])]:
            ce=float(s.mean());row[label]={'ce':ce,'ppl':math.exp(ce)}
        result['rows'][policy]=row
    if not result['missing']:
        fixed=np.load(root/'results/fixed/losses.npy');virtual=np.load(root/'results/frozen/losses.npy')
        result['fixed_vs_frozen_max_token_ce_error']=float(np.max(np.abs(fixed-virtual)))
        assert result['fixed_vs_frozen_max_token_ce_error']<.003
        get=lambda k:result['rows'][k]['post_eviction']['ce']
        result['contrasts']={k+'_minus_natural_ce':get(k)-get('natural') for k in ['frozen','trained','zero_value']}
        result['contrasts']['training_gain_ce']=get('frozen')-get('trained')
        result['contrasts']['value_content_gain_ce']=get('zero_value')-get('trained')
    write(root/('INCOMPLETE.json' if result['missing'] else 'complete.json'),result)
    if result['missing']:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train','evaluate','report']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if a.mode=='prepare':prepare(a.root)
    elif a.mode=='train':train(a.root)
    elif a.mode=='evaluate':evaluate(a.root,int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:report(a.root)


if __name__=='__main__':main()
