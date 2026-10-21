"""Small read-only reset scores and log collection."""
from bibalex_remote import connect
from deploy_qwen_article_reset import LOCAL,REMOTE

def main():
    job=(LOCAL/'jobs.tsv').read_text().strip().split('\t')[1];c=connect()
    try:
        _,out,_=c.exec_command('date -Is; sacct -j '+job+' --format=JobID,State,ExitCode,Elapsed,Restarts')
        status=out.read().decode();print(status);(LOCAL/'slurm-status.txt').write_text(status)
        with c.open_sftp() as s:
            (LOCAL/'logs').mkdir(exist_ok=True)
            for n in s.listdir(REMOTE+'/logs'):
                s.get(REMOTE+'/logs/'+n,str(LOCAL/'logs'/n))
                print(n+'\n'+'\n'.join((LOCAL/'logs'/n).read_text().splitlines()[-5:]))
            for n in ('scores.jsonl','complete.json'):
                try:s.get(REMOTE+'/'+n,str(LOCAL/n))
                except FileNotFoundError:pass
    finally:c.close()

if __name__=='__main__':main()
