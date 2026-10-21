"""Read-only handoff evidence collector; checkpoint files stay on the cluster."""
import json
from bibalex_remote import connect,execute
from deploy_qwen05_optimized_handoff import LOCAL,REMOTE,TRAINING


def main():
    jobs=dict(line.split('\t') for line in (LOCAL/'jobs.tsv').read_text().splitlines())
    client=connect()
    try:
        execute(client,'date -Is; sacct -j 217102,217103,217106,'+','.join(jobs.values())+' --format=JobID,State,ExitCode,Elapsed,Restarts')
        with client.open_sftp() as sftp:
            for name in ('allocation-primary.txt','allocation-continuation.txt','preflight-verified.json','stop-requested.json',
                         'handoff-verified.json','continuation-verified.json'):
                try:sftp.stat(REMOTE+'/'+name)
                except FileNotFoundError:continue
                sftp.get(REMOTE+'/'+name,str(LOCAL/name))
                if name.endswith('.json'):print(name+': '+(LOCAL/name).read_text())
            (LOCAL/'logs').mkdir(exist_ok=True)
            for name in sftp.listdir(TRAINING+'/logs'):
                if name.startswith('ordinary-optimized-'):
                    sftp.get(TRAINING+'/logs/'+name,str(LOCAL/'logs'/name))
                    print(name+':\n'+'\n'.join((LOCAL/'logs'/name).read_text().splitlines()[-6:]))
            with sftp.open(TRAINING+'/runs/ordinary/metrics.jsonl') as stream:
                raw=stream.read().decode()
            (LOCAL/'metrics-observation.jsonl').write_text(raw)
            lines=raw.splitlines()
            if raw and not raw.endswith('\n'):lines=lines[:-1]
            rows=[json.loads(s) for s in lines]
            interesting=[r for r in rows if r['event'] in ('train','startup','paused','implementation_change')]
            print('LATEST_EVENTS '+json.dumps(interesting[-5:],indent=2))
    finally:client.close()


if __name__=='__main__':main()
