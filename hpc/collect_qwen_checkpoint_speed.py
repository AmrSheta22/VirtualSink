"""Collect only small benchmark telemetry; never download checkpoint tensors."""
from bibalex_remote import connect, execute
from deploy_qwen_checkpoint_speed import LOCAL, REMOTE


def main():
    job=(LOCAL/'jobs.tsv').read_text().strip().split('\t')[1]
    client=connect()
    try:
        execute(client,f'sacct -j {job} --format=JobID,State,ExitCode,Elapsed,Restarts; tail -n 5 {REMOTE}/logs/*.out; tail -n 15 {REMOTE}/logs/*.err')
        with client.open_sftp() as sftp:
            (LOCAL/'logs').mkdir(exist_ok=True)
            for name in sftp.listdir(REMOTE+'/logs'):
                sftp.get(REMOTE+'/logs/'+name,str(LOCAL/'logs'/name))
            for name in ('checkpoint-info.json','current.json','optimized.json','comparison.json','metrics.jsonl','slurm-allocation.txt','gpu-before.csv','gpu-after.csv'):
                try:
                    sftp.stat(REMOTE+'/'+name)
                except FileNotFoundError:
                    continue
                sftp.get(REMOTE+'/'+name,str(LOCAL/name))
    finally:
        client.close()
    if (LOCAL/'metrics.jsonl').exists():
        print('LATEST_TRAINING_UPDATES')
        print('\n'.join((LOCAL/'metrics.jsonl').read_text().splitlines()[-3:]))
    if (LOCAL/'comparison.json').exists():
        print((LOCAL/'comparison.json').read_text())


if __name__ == '__main__':
    main()
