"""One-article causal reset probe; retain physical K/V and absolute position."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.recurrent_data import TokenDocuments
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM
from virtualsink.uptraining.train_recurrent import load_checkpoint,token_losses
from virtualsink.uptraining.long_retrieval import clear_memory

RESETS=(8192,11264,11776,12288,12544)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available();torch.set_num_threads(4)
    m=json.loads((args.training_root/'data/manifest.json').read_text())
    config=json.loads((Path(m['model_path'])/'config.json').read_text())
    saved=load_checkpoint(args.root/'ordinary-best.pt');assert saved['step']==2048 and saved['identity']['arm']=='ordinary'
    model=OptimizedRecurrentSinkLM(NativeLM(config,32),128);model.load_state_dict(saved['model']);del saved
    model.cuda().eval();model.checkpointing=False
    data=TokenDocuments(args.training_root/'data','test',model.base.config.vocab_size)
    doc=25;limit=data.length(doc)-1;assert limit==13213
    path=args.root/'scores.jsonl';assert not path.exists();started=time.monotonic()
    snapshots={};state=None;baseline=[]
    def score(start,state,condition,stop):
        losses_all=[]
        for left in range(start,stop,256):
            inputs,targets=data.segment(doc,left,min(256,stop-left))
            out=model(inputs.cuda(),state);state=out.state
            losses=token_losses(model,out.hidden,targets.cuda()).double().cpu()
            assert torch.isfinite(losses).all()
            row=dict(condition=condition,start=left,tokens=len(losses),losses=losses.tolist(),
                loss_sum=float(losses.sum()),elapsed_seconds=time.monotonic()-started)
            with path.open('a') as f:f.write(json.dumps(row)+'\n')
            losses_all.extend(row['losses'])
        return state,losses_all
    for start in range(0,limit,256):
        if start in RESETS:snapshots[start]=state
        state,loss=score(start,state,'intact',min(start+256,limit));baseline.extend(loss)
        if (start+len(loss))%4096==0:print('BASELINE_PROGRESS '+str(start+len(loss)),flush=True)
    results={}
    for boundary in RESETS:
        original=snapshots[boundary];cleared=clear_memory(original)
        assert cleared.position==original.position
        before=[(s.cell.clone(),s.exposed.clone(),s.key.clone(),s.value.clone()) for s in original.layers]
        for old,new in zip(original.layers,cleared.layers):
            assert old.key.data_ptr()==new.key.data_ptr() and old.value.data_ptr()==new.value.data_ptr()
            assert torch.count_nonzero(new.cell)==0 and torch.count_nonzero(new.exposed)==0
        _,loss=score(boundary,cleared,'reset_'+str(boundary),limit)
        for s,old in zip(original.layers,before):
            assert all(torch.equal(x,y) for x,y in zip((s.cell,s.exposed,s.key,s.value),old))
        results[str(boundary)]=dict(tokens=len(loss),reset_ce=sum(loss)/len(loss),
            intact_ce=sum(baseline[boundary:])/len(loss),physical_cache_unchanged=True,
            original_state_not_mutated=True)
        print('RESET_COMPLETE '+json.dumps(dict(boundary=boundary,**results[str(boundary)])),flush=True)
    result=dict(status='complete',job=os.environ['SLURM_JOB_ID'],checkpoint_step=2048,document=doc,
        document_predictions=limit,reset_results=results,elapsed_seconds=time.monotonic()-started,
        scored_predictions=limit+sum(limit-b for b in RESETS))
    (args.root/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
