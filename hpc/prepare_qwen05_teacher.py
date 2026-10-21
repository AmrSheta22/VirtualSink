"""Prepare explicit, official-split WikiText row excerpts using pinned assets."""
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from virtualsink.uptraining.data import sha256


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root = Path(os.environ['QWEN_TEACHER_ROOT'])
    source = Path.home() / 'data/phase3-k80-20260909'
    old = json.loads((source / 'data/qwen05/manifest.json').read_text())
    assert old['model'] == 'Qwen/Qwen2.5-0.5B'
    assert old['revision'] == '060db6499f32faf8b98477b0a26969ef7d8b9987'
    model_path = Path(old['model_path'])
    config = json.loads((model_path / 'config.json').read_text())
    tokenizer = Tokenizer.from_file(str(model_path / 'tokenizer.json'))
    corpus = source / 'hf-cache/hub/datasets--Salesforce--wikitext/snapshots' / old['source']['revision']
    destination = root / 'data'
    destination.mkdir(exist_ok=False)
    manifest = {'model': old['model'], 'revision': old['revision'], 'model_path': str(model_path),
        'source': old['source'], 'format': 'independent-row-excerpts-v1',
        'policy': 'first 129 tokens of distinct nonempty official-split rows with at least 129 tokens; reset every example',
        'model_sha256': {p.name: sha256(p) for p in model_path.iterdir() if p.suffix == '.safetensors' or p.name in ('config.json','tokenizer.json')},
        'config': config, 'splits': {}}
    seen = set()
    for split, count in [('train', 64), ('validation', 8), ('test', 8)]:
        rows, indices, files = [], [], []
        for entry in old['splits'][split]['source_files']:
            path = corpus / entry['file']
            if sha256(path) != entry['sha256']:
                raise ValueError('raw corpus hash mismatch')
            files.append(entry)
            for index, text in enumerate(pq.read_table(path, columns=['text'])['text'].to_pylist()):
                if not text.strip():
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if len(ids) < 129:
                    continue
                excerpt = tuple(ids[:129])
                if excerpt in seen:
                    continue
                seen.add(excerpt)
                rows.append(excerpt)
                indices.append({'file': entry['file'], 'row': index})
                if len(rows) == count:
                    break
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError('insufficient independent excerpts')
        path = destination / (split + '.npy')
        np.save(path, np.asarray(rows, dtype=np.int64), allow_pickle=False)
        manifest['splits'][split] = {'file': path.name, 'examples': count, 'sha256': sha256(path), 'rows': indices, 'source_files': files}
    (root / 'identity.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'event': 'PREPARATION_PASSED', 'model': manifest['model'], 'revision': manifest['revision'], 'examples': {s:v['examples'] for s,v in manifest['splits'].items()}}), flush=True)


if __name__ == '__main__':
    main()
