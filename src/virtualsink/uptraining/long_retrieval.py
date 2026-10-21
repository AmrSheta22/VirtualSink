"""Controlled counterfactual retrieval inputs and scoring (no training)."""
import hashlib
import json
import math
import random
import re

import torch
from .recurrent_sink import LayerMemoryState, MemoryState


GAPS=(0,64,128,256,512,1024,2048,4096,8192,16384)
COLOR_POOL=('red','blue','green','yellow','black','white','orange','pink','purple','brown','gray','silver')
HEADER=('Read each record and answer its question with the color in the record.\n\n'
        'Record: Item 101 has the color red.\nQuestion: What color does item 101 have?\nAnswer: red.\n\n'
        'Record: Item 102 has the color blue.\nQuestion: What color does item 102 have?\nAnswer: blue.\n\n')
PROSE=(
    'The archive contains notes about local roads and public buildings.',
    'A committee reviewed the maintenance schedule during the meeting.',
    'Workers repaired the bridge after inspecting its supporting structure.',
    'The library keeps several reports describing the history of the district.',
    'A visitor followed the path from the station to the museum.',
    'The research team compared measurements from several instruments.',
    'A clerk arranged the documents according to the date of each meeting.',
    'The garden committee discussed a plan for the next growing season.',
    'Several residents requested information about the public transport service.',
    'The engineer explained how the equipment should be checked each month.',
    'The town council discussed repairs to the school and the sports centre.',
    'A report described the journey from the harbour to the railway station.',
)


def build_dataset(encode,seed=20261016):
    rng=random.Random(seed)
    colors=[];ids=[]
    for word in COLOR_POOL:
        tokens=encode(' '+word)
        if len(tokens)==1:
            colors.append(word);ids.append(tokens[0])
        if len(colors)==8:break
    if len(colors)!=8 or len(set(ids))!=8:raise ValueError('need eight distinct single-token color answers')
    missing=next((encode(' '+s)[0] for s in ('unknown','missing','absent') if len(encode(' '+s))==1),None)
    if missing is None or missing in ids:raise ValueError('missing-answer marker must be one noncandidate token')
    keys=rng.sample(list(range(300,900)),8)
    blocks=[]
    for block,key in enumerate(keys):
        distractors='neutral_prose' if block<4 else 'other_item_records'
        local=random.Random(seed+1000+block)
        # Labels never affect the background, question, key or prompt length.
        text='\nUnrelated archive notes follow.\n'
        while len(text)<130000:
            text+=local.choice(PROSE)+'\n'
            if distractors=='other_item_records':
                text+=f'Record: Item {local.randrange(900,1000)} has the color {local.choice(colors)}.\n'
        background=encode(text)[:max(GAPS)]
        if len(background)!=max(GAPS):raise ValueError('insufficient background tokens')
        if distractors=='neutral_prose' and any(re.search(r'\b'+c+r'\b',text,re.I) for c in colors):
            raise ValueError('candidate leakage in neutral background')
        if str(key) in text:raise ValueError('query key leaked into distractors')
        fact_start=encode(f'Record: Item {key} has the color')
        fact_end=encode('.\n')
        query=encode(f'\nQuestion: What color does item {key} have?\nAnswer:')
        header=encode(HEADER)
        prefixes=[header+fact_start+[label]+fact_end for label in ids]
        facts=[fact_start+[label]+fact_end for label in ids]
        label_position=len(header)+len(fact_start)
        assert all(sum(a!=b for a,b in zip(prefixes[0],row))==1 for row in prefixes[1:])
        blocks.append(dict(block=block,key=key,distractors=distractors,prefixes=prefixes,facts=facts,
            missing_prefix=header+fact_start+[missing]+fact_end,background=background,query=query,
            label_position=label_position,fact_suffix_tokens=len(fact_end),
            actual_prediction_distances={str(g):len(prefixes[0])+g+len(query)-1-label_position for g in GAPS}))
    for field in ('prefixes','facts'):
        assert len({len(b[field][0]) for b in blocks})==1,'unaligned '+field
    assert len({len(b['query']) for b in blocks})==1,'unaligned queries'
    return dict(format='counterfactual-color-retrieval-v1',seed=seed,gaps=list(GAPS),colors=colors,
        candidate_token_ids=ids,missing_token_id=missing,blocks=blocks,
        chance_choice_accuracy=1/8,cases_per_gap=64,
        design='eight counterfactual values per fixed key/background/question; four neutral and four interfering blocks; no finetuning')


def payload_sha256(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def score_logits(logits,candidates):
    candidate=torch.tensor(candidates,device=logits.device)
    subset=logits.index_select(-1,candidate).float()
    log_choice=subset.log_softmax(-1)
    log_vocab=subset-logits.float().logsumexp(-1,keepdim=True)
    return dict(choice_predictions=log_choice.argmax(-1).cpu().tolist(),
        greedy_token_ids=logits.argmax(-1).cpu().tolist(),
        candidate_log_probs=log_choice.cpu().tolist(),vocab_candidate_log_probs=log_vocab.cpu().tolist())


def clear_memory(state):
    """Keep bounded physical K/V and absolute position; zero only LSTM state."""
    layers=tuple(LayerMemoryState(s.key,s.value,torch.zeros_like(s.cell),torch.zeros_like(s.exposed)) for s in state.layers)
    return MemoryState(layers,state.position,state.capacity)


def score_rows(scored,blocks,dataset,gap,control,elapsed):
    rows=[];i=0
    missing=control=='missing_answer'
    for block in blocks:
        for truth in range(8):
            index=i if missing else i+truth
            distance=block['actual_prediction_distances'][str(gap)]
            if control=='repeat_fact_near_query':distance=block['fact_suffix_tokens']+len(block['query'])
            if missing:distance=None
            row=dict(block=block['block'],key=block['key'],distractors=block['distractors'],
                filler_tokens=gap,actual_prediction_distance=distance,
                control=control,truth_index=truth,truth_color=dataset['colors'][truth],
                predicted_index=scored['choice_predictions'][index],
                greedy_token_id=scored['greedy_token_ids'][index],
                choice_correct=scored['choice_predictions'][index]==truth,
                greedy_correct=scored['greedy_token_ids'][index]==dataset['candidate_token_ids'][truth],
                choice_nll=-scored['candidate_log_probs'][index][truth],
                vocab_nll=-scored['vocab_candidate_log_probs'][index][truth],
                candidate_log_probs=scored['candidate_log_probs'][index],
                batch_elapsed_seconds=elapsed)
            if not all(math.isfinite(v) for v in (row['choice_nll'],row['vocab_nll'])):
                raise FloatingPointError('nonfinite probe score')
            rows.append(row)
        i+=1 if missing else 8
    return rows
