"""Declared secondary query format after the initial Q/A retrieval test."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from virtualsink.uptraining.long_retrieval import payload_sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,required=True);args=parser.parse_args()
    previous=Path('/cluster/users/alex116u1/data/qwen05-long-retrieval-20261016-r1/dataset.json')
    dataset=json.loads(previous.read_text());parent_hash=dataset.pop('sha256')
    assert payload_sha256(dataset)==parent_hash
    manifest=json.loads((args.training_root/'data/manifest.json').read_text())
    tokenizer=Tokenizer.from_file(str(Path(manifest['model_path'])/'tokenizer.json'))
    for block in dataset['blocks']:
        query=tokenizer.encode(f"\nAccording to the record, item {block['key']} has the color",add_special_tokens=False).ids
        block['query']=query
        newline=tokenizer.encode('\n',add_special_tokens=False).ids
        block['facts']=[newline+tokens for tokens in block['facts']]
        block['actual_prediction_distances']={str(g):len(block['prefixes'][0])+g+len(query)-1-block['label_position'] for g in dataset['gaps']}
    assert len({len(b['query']) for b in dataset['blocks']})==1
    assert len({len(b['facts'][0]) for b in dataset['blocks']})==1
    dataset['parent_dataset_sha256']=parent_hash
    dataset['query_format']='cloze-completion; secondary probe declared after original Q/A results'
    dataset['repeat_fact_separator']='explicit newline before repeated fact'
    dataset['sha256']=payload_sha256(dataset)
    (args.root/'dataset.json').write_text(json.dumps(dataset,indent=2))
    print('CLOZE_DATASET_READY '+json.dumps({k:v for k,v in dataset.items() if k!='blocks'}),flush=True)


if __name__=='__main__':main()
