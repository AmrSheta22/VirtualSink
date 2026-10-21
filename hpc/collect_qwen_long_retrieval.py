"""Download only small retrieval scores, dataset, manifests and logs."""
from bibalex_remote import connect,execute
from deploy_qwen_long_retrieval import LOCAL,REMOTE


def main():
    jobs=dict(s.split('\t') for s in (LOCAL/'jobs.tsv').read_text().splitlines())
    client=connect()
    try:
        _,stdout,_=client.exec_command('date -Is; sacct -j '+','.join(jobs.values())+' --format=JobID,State,ExitCode,Elapsed,Restarts')
        status=stdout.read().decode();(LOCAL/'slurm-status.txt').write_text(status);print(status)
        with client.open_sftp() as sftp:
            (LOCAL/'logs').mkdir(exist_ok=True)
            for name in sftp.listdir(REMOTE+'/logs'):
                sftp.get(REMOTE+'/logs/'+name,str(LOCAL/'logs'/name))
                if name.endswith('.err') and (LOCAL/'logs'/name).stat().st_size:
                    print(name+'\n'+(LOCAL/'logs'/name).read_text()[-2500:])
            for name in sftp.listdir(REMOTE):
                if name=='dataset.json' or name.endswith('-complete.json') or name.endswith('.jsonl'):
                    sftp.get(REMOTE+'/'+name,str(LOCAL/name))
                    if name.endswith('-complete.json'):print((LOCAL/name).read_text())
            for arm in ('memory','memory_disabled','forget_zero','fifo'):
                files=list((LOCAL/'logs').glob(arm+'-*.out'))
                if files:
                    lines=files[-1].read_text().splitlines()
                    probes=[s for s in lines if s.startswith('PROBE ')]
                    print(arm+'\n'+'\n'.join(probes[-2:]+lines[-2:]))
    finally:client.close()


if __name__=='__main__':main()
