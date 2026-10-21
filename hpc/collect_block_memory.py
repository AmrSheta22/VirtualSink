"""Read-only collection of the isolated block-memory gate and pilot outputs."""
import argparse
import json
from pathlib import Path

from bibalex_remote import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    name = f'block-memory-qwen05-20261018-r{args.attempt}'
    local = Path(__file__).resolve().parents[1] / 'artifacts' / name
    remote = '/cluster/users/alex116u1/data/' + name
    jobs = dict(row.split('\t') for row in (local / 'jobs.tsv').read_text().splitlines())
    if not all(value.isdigit() for value in jobs.values()):
        raise ValueError('invalid job IDs')
    client = connect()
    try:
        _, out, _ = client.exec_command('sacct -j ' + ','.join(jobs.values()) + ' -P -n -o JobID,State,Elapsed,ExitCode')
        accounting = out.read().decode()
        observation = dict(jobs=jobs, accounting=accounting.splitlines(), outputs={})
        with client.open_sftp() as sftp:
            for arm, job in jobs.items():
                names = [f'logs/{arm}-{job}.out', f'logs/{arm}-{job}.err']
                names += ['gate.json'] if arm == 'gate' else [f'{arm}/train.jsonl', f'{arm}/result.json']
                tails = {}
                for filename in names:
                    destination = local / filename
                    destination.parent.mkdir(exist_ok=True)
                    try:
                        sftp.get(remote + '/' + filename, str(destination))
                        if filename.endswith(('.out', '.err', 'train.jsonl')):
                            tails[filename] = destination.read_text().splitlines()[-6:]
                        else:
                            tails[filename] = json.loads(destination.read_text())
                    except FileNotFoundError:
                        pass
                observation['outputs'][arm] = tails
        (local / 'observation.json').write_text(json.dumps(observation, indent=2))
        print(json.dumps(observation, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
