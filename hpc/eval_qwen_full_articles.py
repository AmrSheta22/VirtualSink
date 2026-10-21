"""Inference-only full held-out articles; no truncation or model selection."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.recurrent_data import TokenDocuments
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM
from virtualsink.uptraining.train_recurrent import load_checkpoint, token_losses

BANDS=((0,4096),(4096,8192),(8192,12288),(12288,16384))


def position_statistics(losses,start):
    """Return additive, token-weighted sums for zero-based input positions."""
    result={}
    for lo,hi in BANDS:
        left=max(0,lo-start);right=min(len(losses),hi-start)
        selected=losses[left:right] if right>left else losses[:0]
        result[f'{lo}-{hi}']=dict(tokens=len(selected),loss_sum=float(selected.sum()))
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--arm',choices=('memory','memory_disabled','fifo'),required=True)
    args=p.parse_args();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4)
    manifest=json.loads((args.training_root/'data/manifest.json').read_text())
    config=json.loads((Path(manifest['model_path'])/'config.json').read_text())
    checkpoint=args.root/('fifo-best.pt' if args.arm=='fifo' else 'ordinary-best.pt')
    saved=load_checkpoint(checkpoint)
    assert saved['step']==2048 and saved['identity']['arm']==('fifo' if args.arm=='fifo' else 'ordinary')
    model=OptimizedRecurrentSinkLM(NativeLM(config,32),128)
    model.load_state_dict(saved['model']);del saved
    model.cuda().eval();model.checkpointing=False
    flags={} if args.arm=='memory' else dict(read_memory=False,read_null=False,write_memory=False)
    path=args.root/(args.arm+'.jsonl');assert not path.exists()
    started=time.monotonic();records=0;total=0
    for split in ('test','validation'):
        data=TokenDocuments(args.training_root/'data',split,model.base.config.vocab_size)
        # Longest first: obtain the previously missing long-context evidence first.
        order=sorted(range(len(data)),key=lambda i:(-data.length(i),i))
        original_ids=[i for i in range(len(data)) if data.length(i)>4096][:4]
        for document in order:
            state=None;limit=data.length(document)-1;doc_started=time.monotonic()
            assert limit<=16384
            for start in range(0,limit,256):
                inputs,targets=data.segment(document,start,min(256,limit-start))
                out=model(inputs.cuda(),state,**flags);state=out.state
                losses=token_losses(model,out.hidden,targets.cuda()).double().cpu()
                assert torch.isfinite(losses).all()
                row=dict(arm=args.arm,split=split,document=document,document_predictions=limit,
                    original_four=document in original_ids,start=start,tokens=len(losses),
                    loss_sum=float(losses.sum()),bands=position_statistics(losses,start),
                    elapsed_seconds=time.monotonic()-started)
                with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
                total+=len(losses);records+=1
                if (start+len(losses))%4096==0 or start+len(losses)==limit:
                    print('PROGRESS '+json.dumps(dict(arm=args.arm,split=split,document=document,
                        position=start+len(losses),limit=limit,total_tokens=total,
                        elapsed_seconds=time.monotonic()-started)),flush=True)
            del state
            print('DOCUMENT_COMPLETE '+json.dumps(dict(split=split,document=document,
                predictions=limit,seconds=time.monotonic()-doc_started)),flush=True)
    result=dict(status='complete',arm=args.arm,checkpoint_step=2048,job=os.environ['SLURM_JOB_ID'],
        tokens=total,rows=records,elapsed_seconds=time.monotonic()-started,
        splits=['test','validation'],flags=flags)
    target=args.root/(args.arm+'-complete.json');tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
    print('FULL_EVALUATION_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
