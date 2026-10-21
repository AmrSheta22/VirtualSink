"""Strict completion report; never infer successful completion from exit code alone."""
import json
import os
from pathlib import Path
from virtualsink.uptraining.recall_experiment import atomic_json

root=Path(os.environ['QWEN_LONG_ROOT']);plan=json.loads((root/'plan.json').read_text());rows={}
for arm in plan['arms']:
    run=root/'runs'/arm
    if not (run/'complete.json').exists() or not (run/'results.json').exists():
        atomic_json(root/'INCOMPLETE.json',{'arm':arm,'reason':'requested training/evaluation not complete'})
        raise SystemExit('INCOMPLETE')
    completion=json.loads((run/'complete.json').read_text());result=json.loads((run/'results.json').read_text())
    if completion['status']!='complete' or completion['step']!=plan['common']['steps']:raise ValueError('budget mismatch')
    if not json.loads((run/'transfer.json').read_text())['backbone_unchanged']:raise ValueError('warmup changed backbone')
    rows[arm]={'completion':completion,**result}
atomic_json(root/'statistics.json',{'status':'complete','runs':rows,'plan':plan})
lines=['# Qwen0.5B long uptraining','','One seed; selected by held-out validation, then evaluated once on test. Lower CE is better.','',
       '| Arm | Selected step | Test CE | Post-eviction CE | Beyond ten windows CE |','|---|---:|---:|---:|---:|']
for arm,row in rows.items():
    metric=row['results']['test']['ordinary']
    values=[metric[k]['ce'] for k in ('all','post_eviction','beyond_10_windows')]
    lines.append('| '+arm+' | '+str(row['selected_step'])+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'complete','arms':list(rows)}),flush=True)
