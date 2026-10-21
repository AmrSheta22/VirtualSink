"""Read-only collection and matched-step speed summary for the two jobs."""
import json
import statistics
from pathlib import Path
from bibalex_remote import connect, execute
from deploy_recurrent_speed import LOCAL, REMOTE


def main():
    client = connect()
    try:
        _, stdout, _ = client.exec_command('scontrol show job -d 217212; scontrol show job -d 217213; sacct -j 217212,217213 --format=JobID,State,ExitCode,Elapsed,NodeList')
        (LOCAL/'slurm-status.txt').write_text(stdout.read().decode())
        execute(client, 'sacct -j 217212,217213 --format=JobID,State,ExitCode,Elapsed,NodeList; '
                f'tail -n 5 {REMOTE}/logs/*.out; tail -n 12 {REMOTE}/logs/*.err')
        with client.open_sftp() as sftp:
            (LOCAL/'logs').mkdir(exist_ok=True)
            for name in sftp.listdir(REMOTE+'/logs'):
                sftp.get(REMOTE+'/logs/'+name, str(LOCAL/'logs'/name))
            for arm in ('current', 'optimized'):
                try:
                    sftp.stat(REMOTE+'/'+arm+'.json')
                except FileNotFoundError:
                    continue
                sftp.get(REMOTE+'/'+arm+'.json', str(LOCAL/(arm+'.json')))
    finally:
        client.close()
    if not all((LOCAL/(arm+'.json')).exists() for arm in ('current','optimized')):
        print('WAITING_FOR_RESULTS')
        return
    current, optimized = [json.loads((LOCAL/(arm+'.json')).read_text()) for arm in ('current','optimized')]
    assert current['config'] == optimized['config']
    assert current['hostname'] == optimized['hostname']
    assert current['parameters'] == optimized['parameters']
    assert current['gpu_parity'] == optimized['gpu_parity'] == 'PASS'
    cs, os = [[s for s in data['samples'] if s['measured']] for data in (current, optimized)]
    assert len(cs) == len(os) == 30
    assert [(s['step'],s['context_position']) for s in cs] == [(s['step'],s['context_position']) for s in os]
    result = dict(current_job=current['job_id'], optimized_job=optimized['job_id'],
        mean_speedup=current['mean_seconds']/optimized['mean_seconds'],
        median_speedup=current['median_seconds']/optimized['median_seconds'],
        time_reduction_percent=100*(1-optimized['mean_seconds']/current['mean_seconds']),
        current_seconds=current['mean_seconds'], optimized_seconds=optimized['mean_seconds'],
        current_tokens_per_second=current['tokens_per_second'], optimized_tokens_per_second=optimized['tokens_per_second'],
        current_peak_mib=current['peak_allocated_bytes']/2**20, optimized_peak_mib=optimized['peak_allocated_bytes']/2**20,
        max_matched_loss_difference=max(abs(a['loss']-b['loss']) for a,b in zip(cs,os)),
        windows=[dict(steps=[cs[i]['step'],cs[i+9]['step']],
            speedup=statistics.mean(s['seconds'] for s in cs[i:i+10])/statistics.mean(s['seconds'] for s in os[i:i+10]))
            for i in (0,10,20)])
    (LOCAL/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
