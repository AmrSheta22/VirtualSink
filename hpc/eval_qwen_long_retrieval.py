"""Batched long retrieval evaluation from frozen checkpoints, under Slurm."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from virtualsink.uptraining.native import NativeLM
from virtualsink.uptraining.recurrent_sink_optimized import OptimizedRecurrentSinkLM
from virtualsink.uptraining.train_recurrent import load_checkpoint
from virtualsink.uptraining.long_retrieval import build_dataset,payload_sha256,score_logits,score_rows,clear_memory


def write_json(path,payload):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)


def prepare(root,training):
    from tokenizers import Tokenizer
    manifest=json.loads((training/'data/manifest.json').read_text())
    tokenizer=Tokenizer.from_file(str(Path(manifest['model_path'])/'tokenizer.json'))
    dataset=build_dataset(lambda s:tokenizer.encode(s,add_special_tokens=False).ids)
    dataset['sha256']=payload_sha256(dataset)
    write_json(root/'dataset.json',dataset)
    print('DATASET_READY '+json.dumps({k:v for k,v in dataset.items() if k!='blocks'}),flush=True)
    for b in dataset['blocks']:
        print(json.dumps({'block':b['block'],'key':b['key'],'distractors':b['distractors'],
            'prefix_tokens':len(b['prefixes'][0]),'query_tokens':len(b['query']),
            'actual_prediction_distances':b['actual_prediction_distances']}),flush=True)


@torch.no_grad()
def evaluate(root,training,arm):
    dataset=json.loads((root/'dataset.json').read_text()); expected=dataset.pop('sha256')
    assert payload_sha256(dataset)==expected
    manifest=json.loads((training/'data/manifest.json').read_text())
    config=json.loads((Path(manifest['model_path'])/'config.json').read_text())
    snapshot=root/('fifo-best.pt' if arm=='fifo' else 'ordinary-best.pt')
    saved=load_checkpoint(snapshot)
    assert saved['step']==2048 and saved['identity']['arm']==('fifo' if arm=='fifo' else 'ordinary')
    model=OptimizedRecurrentSinkLM(NativeLM(config,32),128)
    model.load_state_dict(saved['model']);del saved
    model.cuda().eval();model.checkpointing=False
    flags={} if arm=='memory' else ({'forget_override':0.} if arm=='forget_zero' else
        dict(read_memory=False,read_null=False,write_memory=False))
    path=root/(arm+'.jsonl');assert not path.exists()
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def tensor(rows):return torch.tensor(rows,device='cuda',dtype=torch.long)
    def advance(tokens,state):
        for start in range(0,tokens.shape[1],256):
            state=model(tokens[:,start:start+256],state,**flags).state
        return state
    def emit(rows):
        with path.open('a') as stream:
            for row in rows:
                row['arm']=arm;stream.write(json.dumps(row)+'\n')
        correct=sum(r['choice_correct'] for r in rows)
        print('PROBE '+json.dumps(dict(arm=arm,control=rows[0]['control'],gap=rows[0]['filler_tokens'],
            blocks=sorted(set(r['block'] for r in rows)),correct=correct,total=len(rows),elapsed=time.monotonic()-started)),flush=True)
    def probe(state,query,blocks,gap,control):
        torch.cuda.synchronize();tick=time.monotonic()
        out=model(query,state,**flags)
        scores=score_logits(model.base.lm_head(out.hidden[:,-1]),dataset['candidate_token_ids'])
        torch.cuda.synchronize()
        emit(score_rows(scores,blocks,dataset,gap,control,time.monotonic()-tick))
    # Four blocks x eight counterfactual labels per batch; branch only queries.
    for offset in (0,4):
        blocks=dataset['blocks'][offset:offset+4]
        prefix=tensor([row for b in blocks for row in b['prefixes']])
        background=tensor([b['background'] for b in blocks for _ in range(8)])
        queries=tensor([b['query'] for b in blocks for _ in range(8)])
        facts=tensor([row for b in blocks for row in b['facts']])
        state=advance(prefix,None);previous=0
        for gap in dataset['gaps']:
            if gap>previous:state=advance(background[:,previous:gap],state)
            probe(state,queries,blocks,gap,'retrieval')
            if arm=='memory':probe(clear_memory(state),queries,blocks,gap,'clear_cells_at_query')
            previous=gap
        probe(state,torch.cat((facts,queries),1),blocks,max(dataset['gaps']),'repeat_fact_near_query')
        del state,prefix,background,queries,facts
    # Missing-label prompts contain no ground-truth token. One stream per key;
    # the identical scores are evaluated against all eight balanced labels.
    blocks=dataset['blocks']
    state=advance(tensor([b['missing_prefix'] for b in blocks]),None)
    background=tensor([b['background'] for b in blocks]);queries=tensor([b['query'] for b in blocks]);previous=0
    for gap in dataset['gaps']:
        if gap>previous:state=advance(background[:,previous:gap],state)
        probe(state,queries,blocks,gap,'missing_answer');previous=gap
    expected_rows=64*(len(dataset['gaps'])*(3 if arm=='memory' else 2)+1)
    count=sum(1 for _ in path.open());assert count==expected_rows
    result=dict(status='complete',arm=arm,job=os.environ['SLURM_JOB_ID'],checkpoint_step=2048,
        dataset_sha256=expected,rows=count,elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),flags=flags,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    write_json(root/(arm+'-complete.json'),result)
    print('RETRIEVAL_COMPLETE '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,required=True);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--arm',choices=('memory','memory_disabled','forget_zero','fifo'))
    args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    if args.prepare:prepare(args.root,args.training_root)
    else:
        assert args.arm and torch.cuda.is_available();evaluate(args.root,args.training_root,args.arm)


if __name__=='__main__':main()
