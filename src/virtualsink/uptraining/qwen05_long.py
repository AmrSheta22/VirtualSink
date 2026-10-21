"""Resumable full-parameter Qwen uptraining with distinct teacher stream state."""
import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path
import torch
from .data import sha256
from .native import load_native
from .recall_experiment import atomic_json
from .recurrent_data import TokenDocuments
from .recurrent_sink import RecurrentSinkLM
from .train_recurrent import evaluate, load_checkpoint, pack_state, unpack_state, unroll_loss

COMMON=dict(seed=1729,steps=2048,memory_warmup=128,lr_warmup=64,capacity=128,
    segment=256,unroll_segments=2,base_lr=1e-5,memory_lr=3e-4,timescales=[512.,4096.,16384.],
    initial_mass=1e-4,calibration_docs=4,calibration_tokens=512,loss_chunk=8,
    save_every=64,eval_every=256,validation_docs=1,validation_tokens=2048,final_docs=4,final_tokens=4096)
ARMS=('fifo','ordinary','full_block_teacher')


def backbone_digest(model):
    h=hashlib.sha256()
    for name,p in model.base.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_optimizer(model,arm,adapting,previous=None):
    groups=model.parameter_groups('full' if adapting else 'memory',COMMON['base_lr'],COMMON['memory_lr'])
    if arm=='fifo':
        for p in model.memory.parameters():p.requires_grad_(False)
        for group in groups:group['params']=[p for p in group['params'] if p.requires_grad]
        groups=[g for g in groups if g['params']]
    if not groups:return None
    opt=torch.optim.AdamW(groups,foreach=False)
    if previous is not None:
        for group in opt.param_groups:
            for p in group['params']:
                if p in previous.state:opt.state[p]=previous.state[p]
    return opt


def set_rates(optimizer,step,adapting):
    if optimizer is None:return
    for group in optimizer.param_groups:
        base=group['weight_decay']>0
        start=COMMON['memory_warmup'] if base else 0
        progress=max(0,step-start)
        warm=min(1.,(progress+1)/COMMON['lr_warmup'])
        decay=.1+.9*.5*(1+math.cos(math.pi*progress/(COMMON['steps']-start)))
        group['lr']=(COMMON['base_lr'] if base else COMMON['memory_lr'])*warm*decay


def train_pair(model,data,cursor,states,arm,device,settings=None,perform_backward=True):
    cfg=COMMON if settings is None else settings
    ordinary_flags={} if arm!='fifo' else dict(read_memory=False,read_null=False,write_memory=False)
    values=[];next_states=[];next_cursor=None;count=None
    for index in range(2):
        flags=dict(ordinary_flags)
        if index==1 and arm=='full_block_teacher':flags['memory_teacher']=True
        loss,n,state,end=unroll_loss(model,data,cursor,states[index],cfg['segment'],cfg['unroll_segments'],flags,device,cfg['loss_chunk'])
        if not n:raise ValueError('corpus exhausted before requested budget')
        if count is not None and (n!=count or end!=next_cursor):raise AssertionError('teacher/control cursors diverged')
        count,next_cursor=n,end
        mean=loss/n
        if not torch.isfinite(mean):raise FloatingPointError('nonfinite objective')
        if perform_backward:mean.backward()
        values.append(mean.item());next_states.append(state.detach() if state is not None else None)
        del loss,mean,state
    return values,count,tuple(next_states),next_cursor


def save(path,model,opt,step,cursor,states,tokens,adapting,identity,best,digest):
    tmp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':None if opt is None else opt.state_dict(),
        'step':step,'cursor':cursor,'states':[pack_state(s) for s in states],'input_tokens':tokens,
        'adapting':adapting,'identity':identity,'best':best,'backbone_digest':digest,
        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp)
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-hours',type=float,default=160.)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():parser.error('Slurm CUDA required')
    root=args.root;run=root/('smoke' if args.smoke else 'runs/'+args.arm)
    if (run/'complete.json').exists():
        print('ALREADY_COMPLETE',flush=True);return
    run.mkdir(parents=True,exist_ok=True)
    def emit(event):
        print(json.dumps(event),flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    manifest_path=root/'data/manifest.json';manifest=json.loads(manifest_path.read_text())
    plan=json.loads((root/'plan.json').read_text())
    if plan['common']!=COMMON:raise ValueError('plan mismatch')
    identity={'arm':args.arm,'common':COMMON,'data_sha256':sha256(manifest_path),'sources':plan['source_sha256'],'smoke':args.smoke}
    for name,digest in manifest['pretrained_sha256'].items():
        if sha256(Path(manifest['model_path'])/name)!=digest:raise ValueError('pretrained asset mismatch')
    torch.manual_seed(COMMON['seed']);torch.cuda.manual_seed_all(COMMON['seed'])
    model=RecurrentSinkLM(load_native(manifest['model_path']),COMMON['capacity'],COMMON['timescales']).cuda()
    model.checkpointing=True
    data={s:TokenDocuments(root/'data',s,model.base.config.vocab_size) for s in ('train','validation','test')}
    ids={s:[i for i in range(len(data[s])) if data[s].length(i)>COMMON['final_tokens']][:COMMON['final_docs']] for s in ('validation','test')}
    if any(len(v)!=COMMON['final_docs'] for v in ids.values()):raise ValueError('missing long held-out articles')
    flags={} if args.arm!='fifo' else dict(read_memory=False,read_null=False,write_memory=False)
    step,tokens,cursor,states,adapting,best=0,0,(0,0),(None,None),False,math.inf
    saved=load_checkpoint(run/'last.pt') if (run/'last.pt').exists() else None
    if saved is not None:
        if saved['identity']!=identity:raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(saved['model']);step=saved['step'];tokens=saved['input_tokens'];cursor=tuple(saved['cursor'])
        states=tuple(unpack_state(s,'cuda') for s in saved['states']);adapting=saved['adapting'];best=saved['best'];digest=saved['backbone_digest']
        opt=make_optimizer(model,args.arm,adapting)
        if opt is not None:opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
    else:
        opt=make_optimizer(model,args.arm,False);digest=backbone_digest(model)
        if args.arm!='fifo':
            calibration=model.calibrate([data['train'].segment(i,0,COMMON['calibration_tokens'])[0].cuda() for i in range(COMMON['calibration_docs'])],target_mass=COMMON['initial_mass'])
            atomic_json(run/'calibration.json',calibration)
        initial=evaluate(model,data['validation'],ids['validation'][:1],COMMON['segment'],512 if args.smoke else 1024,flags,'cuda')
        atomic_json(run/'initial.json',initial)
    atomic_json(run/'config.json',identity)
    emit({'event':'startup','step':step,'arm':args.arm,'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name()})
    stopping=[False]
    def stop(*unused):stopping[0]=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    started=time.monotonic();limit=2 if args.smoke else COMMON['steps'];warmup=1 if args.smoke else COMMON['memory_warmup']
    while step<limit and not stopping[0] and time.monotonic()-started<args.max_hours*3600:
        if step==warmup and not adapting:
            if backbone_digest(model)!=digest:raise AssertionError('frozen backbone changed')
            atomic_json(run/'transfer.json',{'step':step,'backbone_unchanged':True})
            opt=make_optimizer(model,args.arm,True,opt);adapting=True
        model.train();tick=time.monotonic()
        if opt is not None:opt.zero_grad(set_to_none=True)
        set_rates(opt,step,adapting)
        # FIFO has no parameters to update during matched memory-only warmup.
        with torch.set_grad_enabled(opt is not None):
            losses,n,states,cursor=train_pair(model,data['train'],cursor,states,args.arm,'cuda',perform_backward=opt is not None)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True) if opt is not None else torch.tensor(0.)
        if opt is not None:opt.step()
        step+=1;tokens+=n;torch.cuda.synchronize()
        emit({'event':'train','step':step,'ce':losses[0],'auxiliary_ce':losses[1],'input_tokens':tokens,
            'cursor':cursor,'context_position':None if states[0] is None else states[0].position,
            'step_seconds':time.monotonic()-tick,'gradient_norm':norm.item(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        if not args.smoke and (step%COMMON['eval_every']==0 or step==limit):
            report=evaluate(model,data['validation'],ids['validation'][:COMMON['validation_docs']],COMMON['segment'],COMMON['validation_tokens'],flags,'cuda')
            atomic_json(run/f'validation-{step:04d}.json',report);emit({'event':'validation','step':step,**report})
            if report['post_eviction']['ce']<best:
                best=report['post_eviction']['ce'];tmp=run/'best.tmp'
                torch.save({'model':model.state_dict(),'step':step,'identity':identity},tmp);tmp.replace(run/'best.pt')
        if step%COMMON['save_every']==0 or args.smoke:
            save(run/'last.pt',model,opt,step,cursor,states,tokens,adapting,identity,best,digest)
    save(run/'last.pt',model,opt,step,cursor,states,tokens,adapting,identity,best,digest)
    if step<limit:
        atomic_json(run/'paused.json',{'step':step,'input_tokens':tokens});emit({'event':'paused','step':step});return
    if args.smoke:
        # Exercise loading of both stream states and optimizer, not just saving.
        restored=load_checkpoint(run/'last.pt')
        if restored['step']!=2 or len(restored['states'])!=2:raise AssertionError('smoke checkpoint mismatch')
        del restored
    else:
        selected=load_checkpoint(run/'best.pt');model.load_state_dict(selected['model']);selected_step=selected['step'];del selected
        final={}
        for split in ('validation','test'):
            final[split]={}
            policies={'ordinary':flags,'memory_disabled':dict(read_memory=False,read_null=False,write_memory=False)}
            if args.arm!='fifo':policies.update(reads_disabled={'read_memory':False},writes_disabled={'write_memory':False},zero_values={'zero_memory_values':True},forget_zero={'forget_override':0.})
            for name,policy in policies.items():
                report=evaluate(model,data[split],ids[split],COMMON['segment'],COMMON['final_tokens'],policy,'cuda')
                final[split][name]=report;atomic_json(run/f'final-{split}-{name}.json',report)
                emit({'event':'final_evaluation','split':split,'policy':name})
        atomic_json(run/'results.json',{'selected_step':selected_step,'results':final})
    atomic_json(run/'complete.json',{'status':'complete','step':step,'input_tokens':tokens,'elapsed_seconds':time.monotonic()-started,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    emit({'event':'complete','step':step,'input_tokens':tokens})


if __name__=='__main__':main()
