"""Checkpoint-verified, single-writer migration to batched recurrent attention."""
import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def record(path, payload):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2))
    tmp.replace(path)


def events(path):
    text=path.read_text()
    lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(s) for s in lines if s.strip()]


def job_state(job):
    text=subprocess.check_output(['sacct','-j',str(job),'-X','-n','-P','--format=State'],universal_newlines=True)
    states=[s.strip().split('|')[0].split()[0].rstrip('+') for s in text.splitlines() if s.strip()]
    if len(states)!=1:
        raise RuntimeError('ambiguous Slurm state: '+repr(states))
    return states[0]


def signal_legacy(root, job):
    found=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            if directory.stat().st_uid!=os.getuid():continue
            command=(directory/'cmdline').read_bytes().decode().split('\0')
            environment=(directory/'environ').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if ('virtualsink.uptraining.qwen05_long' in command and '--root' in command and '--arm' in command
            and command[command.index('--root')+1]==str(root)
            and command[command.index('--arm')+1]=='ordinary'
            and ('SLURM_JOB_ID='+str(job)).encode() in environment):
            found.append(int(directory.name))
    if len(found)!=1:raise RuntimeError('expected exactly one legacy trainer: '+repr(found))
    os.kill(found[0],signal.SIGUSR1)
    print('SIGNALLED_LEGACY '+json.dumps({'pid':found[0],'job':job,'signal':'USR1','time':time.time()}),flush=True)


def verify_checkpoint(path, root, minimum_step):
    import torch
    from virtualsink.uptraining.train_recurrent import load_checkpoint
    torch.set_num_threads(4)
    before=path.stat()
    time.sleep(5)
    after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert after.st_size>1_000_000_000
    saved=load_checkpoint(path)  # Full tensor/optimizer deserialization, CPU only.
    identity=json.loads((root/'runs/ordinary/config.json').read_text())
    assert saved['identity']==identity and identity['arm']=='ordinary'
    assert saved['step']>=minimum_step and saved['step']<2048 and saved['adapting']
    assert saved['optimizer'] and len(saved['states'])==2
    assert len(saved['optimizer']['param_groups'])==2
    assert saved['optimizer']['state'] and saved['input_tokens']>0
    assert len(saved['cursor'])==2
    config=json.loads((Path(json.loads((root/'data/manifest.json').read_text())['model_path'])/'config.json').read_text())
    width=config['hidden_size']//config['num_attention_heads']
    for state in saved['states']:
        if state is None:
            assert saved['cursor'][1]==0
            continue
        assert state['capacity']==128 and state['position']==saved['cursor'][1]
        assert len(state['layers'])==config['num_hidden_layers']
        count=min(state['position'],128)
        for layer in state['layers']:
            assert layer['key'].shape==layer['value'].shape==(1,config['num_key_value_heads'],count,width)
            assert layer['cell'].shape==layer['exposed'].shape==(1,config['num_attention_heads'],3,width)
    tensors=0
    def finite(value):
        nonlocal tensors
        if isinstance(value,torch.Tensor):
            tensors+=1
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('nonfinite checkpoint tensor')
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite(child)
    finite(saved['model']);finite(saved['optimizer']);finite(saved['states'])
    rows=events(root/'runs/ordinary/metrics.jsonl')
    rows=[r for r in rows if r['event']=='train' and r['step']==saved['step']]
    assert len(rows)==1 and rows[0]['input_tokens']==saved['input_tokens'] and list(rows[0]['cursor'])==list(saved['cursor'])
    result=dict(step=saved['step'],input_tokens=saved['input_tokens'],cursor=saved['cursor'],
        stream_positions=[None if s is None else s['position'] for s in saved['states']],
        bytes=after.st_size,inode=after.st_ino,mtime=after.st_mtime,verified_at=time.time(),
        finite_tensors=tensors,optimizer_states=len(saved['optimizer']['state']),
        full_deserialization=True,metadata_stable=True,identity_matches=True)
    del saved
    gc.collect()
    return result


def train(plan, work, resume_step):
    root=Path(plan['training_root'])
    code='from virtualsink.uptraining import qwen05_long as t; from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM; t.RecurrentSinkLM=OptimizedRecurrentSinkLM; print("OPTIMIZED_IMPLEMENTATION_ACTIVE",flush=True); t.main()'
    process=subprocess.Popen([sys.executable,'-u','-c',code,'--root',str(root),'--arm','ordinary','--max-hours','160'])
    def forward(signum,frame):
        if process.poll() is None:process.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,forward)
    signal.signal(signal.SIGTERM,forward)
    deadline=time.monotonic()+1800
    verified=False
    while process.poll() is None:
        if not verified:
            rows=events(root/'runs/ordinary/metrics.jsonl')
            startup=[r for r in rows if r['event']=='startup' and r.get('job')==os.environ['SLURM_JOB_ID']]
            updates=[r for r in rows if r['event']=='train' and r['step']>resume_step]
            if startup:
                assert startup[-1]['step']==resume_step
            if len(updates)>=3:
                assert startup and [r['step'] for r in updates[:3]]==list(range(resume_step+1,resume_step+4))
                assert all(all(math.isfinite(r[k]) for k in ('ce','auxiliary_ce','gradient_norm','step_seconds')) for r in updates[:3])
                assert all(b['input_tokens']>a['input_tokens'] for a,b in zip(updates,updates[1:]))
                proof=dict(status='running_verified',job=os.environ['SLURM_JOB_ID'],implementation='OptimizedRecurrentSinkLM',
                    resumed_step=resume_step,first_updates=updates[:3],verified_at=time.time(),
                    mean_step_seconds=sum(r['step_seconds'] for r in updates[:3])/3,
                    source_manifest=str(work/'source-sha256.json'))
                record(work/'continuation-verified.json',proof)
                record(root/'optimized-continuation.json',proof)
                print('CONTINUATION_VERIFIED '+json.dumps(proof),flush=True)
                verified=True
            if time.monotonic()>deadline and not verified:
                process.send_signal(signal.SIGUSR1)
                raise RuntimeError('continuation failed to produce three finite updates within 30 minutes; graceful stop requested')
        time.sleep(5)
    code=process.wait()
    if code:raise RuntimeError('optimized training exited '+str(code))
    if not verified and not (root/'runs/ordinary/complete.json').exists():
        raise RuntimeError('training exited before verified continuation or completion')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path)
    parser.add_argument('--continuation',action='store_true')
    parser.add_argument('--signal-only',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--legacy-job',type=int)
    args=parser.parse_args()
    if args.signal_only:
        signal_legacy(args.root,args.legacy_job);return
    work=args.work;plan=json.loads((work/'handoff-plan.json').read_text());root=Path(plan['training_root'])
    if (root/'runs/ordinary/complete.json').exists():
        print('ALREADY_COMPLETE',flush=True);return
    if args.continuation:
        audit=verify_checkpoint(root/'runs/ordinary/last.pt',root,plan['minimum_checkpoint_step'])
        train(plan,work,audit['step']);return
    while time.time()<plan['not_before_epoch']:time.sleep(min(10,plan['not_before_epoch']-time.time()))
    assert job_state(plan['legacy_primary'])=='RUNNING'
    jobs=dict(s.split('\t') for s in (work/'jobs.tsv').read_text().splitlines())
    report_job=jobs.get('report-optimized',str(plan['report_job']))
    report_status=subprocess.check_output(['scontrol','show','job',report_job],universal_newlines=True)
    assert 'afterany:'+jobs['ordinary-optimized-1'] in report_status
    os.link(root/'runs/ordinary/last.pt',work/'preflight.pt')
    preflight=verify_checkpoint(work/'preflight.pt',root,plan['minimum_checkpoint_step'])
    record(work/'preflight-verified.json',preflight)
    print('PREFLIGHT_CHECKPOINT_VERIFIED '+json.dumps(preflight),flush=True)
    # Prevent a second writer from starting when the primary exits gracefully.
    queued_state=job_state(plan['legacy_continuation'])
    if queued_state=='PENDING':
        subprocess.run(['scancel',str(plan['legacy_continuation'])],check=True)
    elif not queued_state.startswith('CANCELLED'):
        raise RuntimeError('unexpected legacy continuation state: '+queued_state)
    requested=time.time()
    subprocess.run(['srun','--jobid='+str(plan['legacy_primary']),'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=256M',
        '--nodelist='+plan['legacy_node'],sys.executable,str(Path(__file__).resolve()),'--signal-only',
        '--root',str(root),'--legacy-job',str(plan['legacy_primary'])],check=True,timeout=120)
    record(work/'stop-requested.json',dict(requested_at=requested,legacy_primary=plan['legacy_primary'],cancelled_queued=plan['legacy_continuation']))
    deadline=time.monotonic()+2700
    while job_state(plan['legacy_primary']) in ('RUNNING','COMPLETING'):
        if time.monotonic()>deadline:raise TimeoutError('legacy job did not finish gracefully; no optimized writer started')
        time.sleep(10)
    assert job_state(plan['legacy_primary'])=='COMPLETED'
    paused=json.loads((root/'runs/ordinary/paused.json').read_text())
    assert (root/'runs/ordinary/paused.json').stat().st_mtime>=requested-2
    os.link(root/'runs/ordinary/last.pt',work/'handoff-checkpoint.pt')
    audit=verify_checkpoint(work/'handoff-checkpoint.pt',root,preflight['step'])
    assert audit['step']==paused['step'] and audit['input_tokens']==paused['input_tokens']
    assert audit['mtime']>=requested-2
    migration=dict(status='checkpoint_verified_before_resume',legacy_primary=plan['legacy_primary'],
        legacy_continuation=plan['legacy_continuation'],new_job=os.environ['SLURM_JOB_ID'],
        checkpoint=audit,requested_at=requested,implementation='OptimizedRecurrentSinkLM',
        rollback_checkpoint=str(work/'handoff-checkpoint.pt'),source_manifest=str(work/'source-sha256.json'))
    record(work/'handoff-verified.json',migration)
    record(root/'optimized-handoff.json',migration)
    with (root/'runs/ordinary/metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event='implementation_change',step=audit['step'],implementation='OptimizedRecurrentSinkLM',job=os.environ['SLURM_JOB_ID']))+'\n')
    print('HANDOFF_CHECKPOINT_VERIFIED '+json.dumps(migration),flush=True)
    train(plan,work,audit['step'])


if __name__=='__main__':main()
