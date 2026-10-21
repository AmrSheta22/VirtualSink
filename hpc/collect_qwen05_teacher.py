"""Read-only compact progress and artifact collection; never downloads weights."""
import json
from datetime import datetime, timezone
from pathlib import Path
from bibalex_remote import connect

ROOT=Path(__file__).resolve().parents[1]
NAME='qwen05-full-block-teacher-20261014-r1'
LOCAL=ROOT/'artifacts'/NAME
REMOTE='/cluster/users/alex116u1/data/'+NAME


def main():
    jobs=dict(line.split('\t') for line in (LOCAL/'jobs.tsv').read_text().splitlines())
    if not all(v.isdigit() for v in jobs.values()):
        raise ValueError('invalid job identity')
    observation={'checked_at':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'logs':{},'runs':{}}
    client=connect()
    try:
        _,out,_=client.exec_command('sacct -j '+','.join(jobs.values())+' -P -n -o JobID,State,Elapsed,ExitCode')
        observation['accounting']=out.read().decode().splitlines()
        with client.open_sftp() as sftp:
            def fetch(name):
                try:
                    with sftp.open(REMOTE+'/'+name) as f:
                        data=f.read()
                except FileNotFoundError:
                    return None
                p=LOCAL/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                return data.decode()
            for name in sftp.listdir(REMOTE+'/logs'):
                text=fetch('logs/'+name)
                observation['logs'][name]=text.splitlines()[-5:]
            for name in ('identity.json','statistics.json','REPORT.md','INCOMPLETE.json'):
                fetch(name)
            for arm in ('smoke','runs/ordinary','runs/full_block_teacher'):
                for name in ('config.json','initial.json','transfer.json','validation.json','test.json','complete.json'):
                    fetch(arm+'/'+name)
                text=fetch(arm+'/metrics.jsonl')
                if text:
                    events=[]
                    for line in text.splitlines():
                        try:events.append(json.loads(line))
                        except json.JSONDecodeError:pass
                    observation['runs'][arm]=events[-1] if events else None
    finally:
        client.close()
    (LOCAL/'observation.json').write_text(json.dumps(observation,indent=2))
    compact=dict(observation)
    compact['logs']={n:[line[:700] for line in lines] for n,lines in observation['logs'].items()}
    compact['runs']={n:{k:v for k,v in event.items() if k!='calibration'} for n,event in observation['runs'].items()}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':
    main()
