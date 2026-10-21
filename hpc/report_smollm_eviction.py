"""Read-only cluster status and download completed prefix-replacement results."""
import json
from pathlib import Path
from bibalex_remote import connect, execute

LOCAL = Path(__file__).resolve().parents[1] / 'artifacts/smollm-eviction-20261016'
REMOTE = '/cluster/users/alex116u1/data/smollm-eviction-20261016'


def main():
    client = connect()
    try:
        jobs = ','.join(line.split()[1] for line in (LOCAL / 'jobs.tsv').read_text().splitlines())
        assert all(job.isdigit() for job in jobs.split(','))
        patterns = ' -o '.join(f"-name '*{job}*.err'" for job in jobs.split(','))
        execute(client, f"sacct -j {jobs} --format=JobID,State,ExitCode,Elapsed -X; find {REMOTE}/logs -type f \\( {patterns} \\) -size +0c -exec tail -n 12 {{}} +; find {REMOTE}/results -name progress.json -exec cat {{}} +")
        sftp = client.open_sftp()
        for name in ['identity.json', 'aggregate.json', 'INCOMPLETE.json']:
            try:
                sftp.stat(REMOTE + '/' + name)
                sftp.get(REMOTE + '/' + name, str(LOCAL / name))
                print('DOWNLOADED', name)
            except FileNotFoundError:
                pass
        if (LOCAL / 'aggregate.json').exists() and (LOCAL / 'aggregate.json').stat().st_size:
            result = json.loads((LOCAL / 'aggregate.json').read_text())
            print(json.dumps(dict(complete=result['complete'],
                seeds=[dict(seed=s['seed'], contrasts=s['contrasts']) for s in result['seeds']]), indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
