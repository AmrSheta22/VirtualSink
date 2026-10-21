"""Read-only loss/checkpoint review of collected Qwen long-run telemetry."""
import json
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from bibalex_remote import connect

ROOT=Path(__file__).resolve().parents[1]/'artifacts/qwen05-long-uptraining-20261014-r1'


def main():
    now=datetime.now(ZoneInfo('Africa/Cairo'));tag=now.strftime('%Y%m%d-%H%M')
    plan=json.loads((ROOT/'plan.json').read_text());remote=plan['remote_root']
    stop_path=ROOT/'teacher-stop.json'
    active_arms=[a for a in plan['arms'] if not (stop_path.exists() and a=='full_block_teacher')]
    trains={};events={}
    for arm in active_arms:
        events[arm]=[json.loads(s) for s in (ROOT/'runs'/arm/'metrics.jsonl').read_text().splitlines()]
        trains[arm]=[e for e in events[arm] if e['event']=='train']
        if [e['step'] for e in trains[arm]]!=list(range(1,len(trains[arm])+1)):
            raise ValueError('noncontiguous/replayed logs require manual reconciliation')
    common=min(len(rows) for rows in trains.values())
    for i in range(common):
        if len({(rows[i]['input_tokens'],tuple(rows[i]['cursor']),rows[i]['context_position']) for rows in trains.values()})!=1:
            raise ValueError('matched data cursors differ')
    windows=[(1,32),(97,128)]+[(lo,min(lo+63,common)) for lo in range(129,common+1,64) if min(64,common-lo+1)>=16]
    report={'checked_at_cairo':now.isoformat(),'matched_steps_verified':common,'runs':{},
            'teacher_stop':json.loads(stop_path.read_text()) if stop_path.exists() else None}
    client=connect()
    try:
        ids=','.join(dict(s.split('\t') for s in (ROOT/'jobs.tsv').read_text().splitlines()).values())
        _,out,_=client.exec_command('sacct -j '+ids+' --format=JobID,State,Elapsed,Restarts -P')
        report['accounting']=out.read().decode().splitlines()
        with client.open_sftp() as sftp:
            for arm,rows in trains.items():
                last=rows[-1];recent=rows[-64:]
                changes=[e for e in events[arm] if e['event']=='implementation_change']
                if changes:
                    recent=[e for e in recent if e['step']>changes[-1]['step']]
                if not recent:recent=rows[-1:]
                rate=statistics.mean(e['step_seconds'] for e in recent)
                d={k:last[k] for k in ('step','input_tokens','context_position')}
                d.update(recent_seconds_per_step=rate,phase='full backbone' if last['step']>128 else 'memory warmup',
                    nonfinite_records=sum(not all(math.isfinite(e[k]) for k in ('ce','auxiliary_ce','gradient_norm')) for e in rows),
                    peak_reserved_GB=max(e['peak_reserved_bytes'] for e in rows)/1e9,
                    validation=[e for e in events[arm] if e['event']=='validation'],windows=[])
                if changes:
                    d.update(implementation=changes[-1]['implementation'],implementation_changed_at_step=changes[-1]['step'],throughput_sample_steps=len(recent))
                for filename in ('complete.json','results.json'):
                    path=ROOT/'runs'/arm/filename
                    if path.exists():d[filename[:-5]]=json.loads(path.read_text())
                if 'complete' in d:d['phase']='complete'
                for lo,hi in windows:
                    segment=rows[lo-1:hi]
                    if len(segment)!=hi-lo+1:continue
                    weights=[e['input_tokens']-(rows[e['step']-2]['input_tokens'] if e['step']>1 else 0) for e in segment]
                    d['windows'].append({'steps':[lo,hi],**{k:sum(e[k]*n for e,n in zip(segment,weights))/sum(weights) for k in ('ce','auxiliary_ce')},
                        'gradient_median':statistics.median(e['gradient_norm'] for e in segment),
                        'gradient_max':max(e['gradient_norm'] for e in segment)})
                rp=remote+'/runs/'+arm;d['checkpoints']={}
                for name in ('last.pt','best.pt'):
                    try:st=sftp.stat(rp+'/'+name)
                    except FileNotFoundError:continue
                    at=datetime.fromtimestamp(st.st_mtime,ZoneInfo('Africa/Cairo'))
                    d['checkpoints'][name]={'bytes':st.st_size,'written_at_cairo':at.isoformat(),'age_minutes':(now-at).total_seconds()/60}
                for name in sftp.listdir(rp):
                    if name.startswith('validation-') and name.endswith('.json'):
                        sftp.get(rp+'/'+name,str(ROOT/'runs'/arm/name))
                if d['phase']=='complete':
                    at=datetime.fromtimestamp(sftp.stat(rp+'/complete.json').st_mtime,ZoneInfo('Africa/Cairo'))
                    d['completed_at_cairo']=at.isoformat()
                    report['runs'][arm]=d
                    continue
                at=datetime.fromtimestamp(sftp.stat(rp+'/metrics.jsonl').st_mtime,ZoneInfo('Africa/Cairo'))
                remaining=(2048-last['step'])*rate
                overhead=(8-last['step']//256)*240+((2048-last['step'])//64)*30+(3600 if arm=='fifo' else 14400)
                d['estimated_training_finish_cairo']=(at+timedelta(seconds=remaining)).isoformat()
                d['estimated_finish_with_overhead_cairo']=(at+timedelta(seconds=remaining+overhead)).isoformat()
                report['runs'][arm]=d
    finally:client.close()
    (ROOT/('monitor-'+tag+'.json')).write_text(json.dumps(report,indent=2))
    (ROOT/('observation-'+tag+'.json')).write_text((ROOT/'observation.json').read_text())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
