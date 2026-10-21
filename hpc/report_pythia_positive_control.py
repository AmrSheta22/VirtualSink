"""Read-only status and result collector for the Pythia replacement control."""
import json
from pathlib import Path
from bibalex_remote import connect, execute

LOCAL = Path(__file__).resolve().parents[1]/'artifacts/pythia-positive-control-20261017'
REMOTE = '/cluster/users/alex116u1/data/pythia-positive-control-20261017'


def main():
    jobs = [line.split()[1] for line in (LOCAL/'jobs.tsv').read_text().splitlines()]
    assert all(job.isdigit() for job in jobs)
    client = connect()
    try:
        execute(client, f"sacct -j {','.join(jobs)} --format=JobID,State,ExitCode,Elapsed -X; find {REMOTE}/logs -name '*.err' -size +0c -exec tail -n 12 {{}} +; find {REMOTE}/results -name progress.json -exec cat {{}} +")
        with client.open_sftp() as sftp:
            for name in ['identity.json','aggregate.json','INCOMPLETE.json']:
                try:
                    sftp.stat(REMOTE+'/'+name)
                    sftp.get(REMOTE+'/'+name,str(LOCAL/name))
                    print('DOWNLOADED',name)
                except FileNotFoundError:
                    continue
        if (LOCAL/'aggregate.json').exists():
            result = json.loads((LOCAL/'aggregate.json').read_text())
            print(json.dumps(dict(complete=result['complete'],contrasts=result.get('contrasts')),indent=2))
    finally: client.close()


if __name__=='__main__': main()
