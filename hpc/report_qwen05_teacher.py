"""Strict tiny-screen result collection on Slurm CPU."""
import json
import os
from pathlib import Path
from virtualsink.uptraining.recall_experiment import atomic_json

root=Path(os.environ['QWEN_TEACHER_ROOT'])
plan=json.loads((root/'plan.json').read_text())
rows={}
for arm in plan['arms']:
    run=root/'runs'/arm
    paths=[run/n for n in ('complete.json','initial.json','validation.json','test.json','transfer.json')]
    if not all(p.exists() for p in paths):
        atomic_json(root/'INCOMPLETE.json',{'arm':arm,'missing':[str(p) for p in paths if not p.exists()]})
        raise SystemExit('incomplete experiment')
    complete,initial,validation,test,transfer=[json.loads(p.read_text()) for p in paths]
    if complete['status']!='completed' or complete['steps']!=plan['common']['steps'] or not transfer['backbone_unchanged']:
        raise ValueError('budget or frozen backbone mismatch')
    rows[arm]={'initial':initial,'validation':validation,'test':test,'completion':complete}
atomic_json(root/'statistics.json',{'status':'complete','runs':rows,'plan':plan})
lines=['# Qwen0.5B full-block-teacher feasibility screen','','One seed; 8192 unique training tokens per arm. No definitive language-model gain claim.','',
       '| Arm | Initial validation CE | Final validation CE | Test CE | Test post-eviction CE | Test reads disabled CE |','|---|---:|---:|---:|---:|---:|']
for arm,r in rows.items():
    values=[r['initial']['ordinary']['ce'],r['validation']['ordinary']['ce'],r['test']['ordinary']['ce'],r['test']['ordinary']['post_eviction_ce'],r['test']['reads_disabled']['ce']]
    lines.append('| '+arm+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'complete','arms':list(rows)}),flush=True)
