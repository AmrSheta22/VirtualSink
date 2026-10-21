"""Read-only full-article evaluation collection; no checkpoint downloads."""
from bibalex_remote import connect
from deploy_qwen_full_articles import LOCAL,REMOTE,ARMS


def main():
    jobs=dict(line.split('\t') for line in (LOCAL/'jobs.tsv').read_text().splitlines())
    c=connect()
    try:
        _,out,_=c.exec_command('date -Is; sacct -j '+','.join(jobs.values())+' --format=JobID,State,ExitCode,Elapsed,Restarts')
        status=out.read().decode();print(status);(LOCAL/'slurm-status.txt').write_text(status)
        with c.open_sftp() as s:
            (LOCAL/'logs').mkdir(exist_ok=True)
            for name in s.listdir(REMOTE+'/logs'):
                s.get(REMOTE+'/logs/'+name,str(LOCAL/'logs'/name))
                lines=(LOCAL/'logs'/name).read_text().splitlines()
                if name.endswith('.err') and lines:print(name+'\n'+'\n'.join(lines[-15:]))
                if name.endswith('.out'):
                    events=[line for line in lines if line.startswith(('PROGRESS','DOCUMENT_COMPLETE','FULL_EVALUATION_COMPLETE'))]
                    print(name+'\n'+'\n'.join(events[-2:]+lines[-2:]))
            for name in s.listdir(REMOTE):
                if name.endswith('.jsonl') or name.endswith('-complete.json'):
                    s.get(REMOTE+'/'+name,str(LOCAL/name))
    finally:c.close()


if __name__=='__main__':main()
