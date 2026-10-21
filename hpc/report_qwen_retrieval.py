"""Local two-format retrieval report and scientific accuracy figure."""
import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=('qwen05-long-retrieval-20261016-r1','qwen05-long-retrieval-cloze-20261016-r1')
ARMS=('memory','memory_disabled','forget_zero','fifo')
NAMES={'memory':'Memory enabled','memory_disabled':'Same weights, memory disabled','forget_zero':'Direct forget factor = 0','fifo':'Trained FIFO'}


def main():
    runs=[]
    audits=[]
    for experiment in EXPERIMENTS:
        path=ROOT/'artifacts'/experiment
        analysis=json.loads((path/'analysis.json').read_text());assert analysis['complete']
        dataset=json.loads((path/'dataset.json').read_text())
        hashes=json.loads((path/'source-sha256.json').read_text())
        with tarfile.open(path/'release.tar.gz') as archive:
            assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest for name,digest in hashes.items())
        for arm in ARMS:
            logs=list((path/'logs').glob(arm+'-*.err'));assert len(logs)==1 and logs[0].stat().st_size==0
            assert analysis['arms'][arm]['completion']['rows']==(1984 if arm=='memory' else 1344)
            job=analysis['arms'][arm]['completion']['job']
            records=[line.split() for line in (path/'slurm-status.txt').read_text().splitlines() if line.split() and line.split()[0]==job]
            assert len(records)==1 and records[0][1:3]==['COMPLETED','0:0'] and records[0][-1]=='0'
        audits.append(dict(experiment=experiment,dataset_sha256=dataset['sha256'],
            frozen_source_hashes_verified=True,error_logs_empty=True,all_jobs_completed_exit_zero_no_restarts=True,
            arms={arm:analysis['arms'][arm]['completion'] for arm in ARMS}))
        runs.append((path,analysis,dataset))
    target=runs[0][0]
    (target/'final-retrieval-audit.json').write_text(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
        evaluations=audits,checkpoints='Unchanged validation-selected step2048, retained on HPC',
        dataset_pair=json.loads((runs[1][0]/'dataset-pair-audit.json').read_text())),indent=2))
    colors=('#7040a0','#3787a5','#cf7b26','#282d35')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,(_,analysis,data),title in zip(axes,runs,('Original question / answer','Secondary completion-style query')):
        x=[data['blocks'][0]['actual_prediction_distances'][str(g)] for g in data['gaps']]
        ax.axvspan(128,22000,color='#f2f3f5',zorder=0)
        for arm,color,marker in zip(ARMS,colors,('o','s','^','x')):
            y=[analysis['arms'][arm]['metrics']['retrieval'][str(g)]['choice_accuracy'] for g in data['gaps']]
            ax.plot(x,y,marker=marker,color=color,label=NAMES[arm],linewidth=1.7,markersize=5,alpha=.9)
        ax.axhline(.125,color='#89909a',linestyle=':',linewidth=1.2)
        ax.axvline(128,color='#a6abb2',linestyle='--',linewidth=.8)
        ax.set_xscale('log',base=2);ax.set_xlim(12,22000);ax.set_ylim(0,1.06)
        ax.set_xticks((16,128,1024,16384),('16','128','1,024','16,384'))
        ax.set_yticks((0,.125,.25,.5,.75,1));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.set_title(title,fontsize=11);ax.set_xlabel('Source-answer to prediction distance (tokens)',fontsize=9)
        ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.text(.97,.94,'Shaded: outside physical cache',ha='right',va='top',transform=ax.transAxes,fontsize=8,color='#666b73')
        ax.annotate('Chance 12.5%',xy=(14500,.125),xytext=(13500,.19),ha='right',fontsize=8,color='#636b75')
    axes[0].set_ylabel('Eight-choice accuracy')
    fig.suptitle('Controlled fact retrieval — trained Qwen0.5B checkpoints',fontsize=14,y=.99)
    fig.text(.5,.92,'64 balanced counterfactual cases per distance; eight key/background blocks',ha='center',fontsize=9,color='#555b64')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False,fontsize=8.5)
    fig.tight_layout(rect=(0,.09,1,.9))
    fig.savefig(target/'retrieval-accuracy.png',dpi=160,bbox_inches='tight')
    fig.savefig(target/'retrieval-accuracy.svg',bbox_inches='tight');plt.close(fig)
    lines=['# Controlled long-memory retrieval — final report','',
        'Evaluation only: completed validation-selected step2048 memory and FIFO checkpoints, FP32 K80, physical capacity128, segment256, batch32. No training, calibration or checkpoint changes. Four arms completed successfully in each of two separately frozen query formats.','',
        'The original Q/A test was primary. After its long-context generation/control limitations became apparent, a separately labelled completion-format follow-up used exactly the same facts, keys and backgrounds, changed only the query template, and inserted an explicit newline before the repeated-fact control. This follow-up is exploratory, not a preregistered independent replication.','',
        '![Retrieval accuracy](retrieval-accuracy.png)','',
        '## Primary endpoint: balanced eight-color choice accuracy','',
        'Each distance contains64 cases from eight keys/backgrounds. Within each block, only the source color token changes across all eight values. Four backgrounds are neutral prose, four contain other item/color records. Chance is12.5%. Candidate answer tokens are never fed after the query. Query branches are discarded before continuing background.','']
    for (_,analysis,data),title in zip(runs,('Original Q/A format','Completion-format follow-up')):
        lines+=['### '+title,'','| Filler tokens | Actual source distance | Memory | Disabled, same weights | Forget=0 | FIFO |','|---:|---:|---:|---:|---:|---:|']
        for gap in data['gaps']:
            distance=data['blocks'][0]['actual_prediction_distances'][str(gap)]
            cells=[f"{analysis['arms'][arm]['metrics']['retrieval'][str(gap)]['choice_accuracy']:.1%}" for arm in ARMS]
            lines.append('| '+str(gap)+' | '+str(distance)+' | '+' | '.join(cells)+' |')
        lines+=['','| Control | Memory | Disabled | Forget=0 | FIFO |','|---|---:|---:|---:|---:|']
        for control,gap,label in (('retrieval',0,'Nearby fact at start'),('repeat_fact_near_query',16384,'Fact repeated near query after16K filler'),('missing_answer',16384,'Missing-answer control after16K filler')):
            lines.append('| '+label+' | '+' | '.join(f"{analysis['arms'][arm]['metrics'][control][str(gap)]['choice_accuracy']:.1%}" for arm in ARMS)+' |')
        lines+=['','Exact first-token color accuracy (case/format sensitive) for repeated-fact control: '+', '.join(NAMES[arm]+' '+f"{analysis['arms'][arm]['metrics']['repeat_fact_near_query']['16384']['greedy_accuracy']:.1%}" for arm in ARMS)+'.','']
    cloze=runs[1][1]['arms']['memory']['metrics']
    assert all(cloze['retrieval'][str(g)]['correct']==8 for g in runs[1][2]['gaps'] if g>=128)
    lines+=['## Interpretation and limits','',
        'In the completion-format follow-up, memory answered 64/64 nearby facts, 47/64 after 64 filler tokens, and 8/64 at every tested gap from 128 through 16,384 filler tokens. Repeating the fact near the final query restored 57/64 answers; disabling memory on the same weights gave 60/64. The long-position positive control is therefore strong but imperfect. The source-to-prediction distance at first chance performance is 143 tokens, versus physical cache capacity 128.', '',
        'The tested memory does not demonstrate reliable arbitrary-fact retrieval beyond the physical cache on these formats. Nearby facts validate source/label/scoring alignment. Every missing-answer control scores exactly12.5%, and FIFO/fully-disabled predictions become exactly invariant to source color beyond their finite24*(128-1)=3048-token input receptive field. Those checks passed.','',
        'Do not equate small logit sensitivity with correct retrieval: memory-enabled predictions retain tiny counterfactual variations at long distances while answer accuracy can remain at chance. Raw per-distance association scores and descriptive stratified block-bootstrap intervals are in analysis.json for each format. Clearing cells at query preserves physical K/V, which may already encode prior memory effects. Forget=0 removes direct cell retention but leaves recurrent gate projections.','',
        'The primary score ranks eight lowercase single-token color answers. Auxiliary unrestricted scoring measures only the exact first token, not complete free-form generations; a model starting a sentence can fail this formatting-sensitive metric despite ranking the color well. The first repeated-fact control also omitted a separating newline, corrected in the labelled completion follow-up. Long-position positive controls are imperfect and must be reported numerically, not assumed successful.','',
        'Eight independent key/background blocks are a small sample. Counterfactual labels and repeated distances are dependent, bootstrap intervals across ten distances are exploratory/unadjusted, and both formats reuse facts/backgrounds. One trained checkpoint, synthetic prose and other-key distractors, no full-attention long-context upper bound and no multi-token generation test limit broader claims. These tests cannot isolate encoding, addressing, interference and prompt-format failure. They do not prove the architecture could never learn retrieval under direct supervision.','',
        'The earlier WikiText ablations showed memory helping the memory-trained model while FIFO retained slightly better test CE. Such language-model benefits do not establish arbitrary-fact recall. The controlled results should be used alongside that distinction.','',
        'Raw artifacts: dataset.json, plan.json, per-arm JSONL scores, per-arm completion records, source-sha256.json, frozen release.tar.gz, slurm-status.txt, logs and analysis.json in each experiment directory. Model checkpoints remained on the cluster.']
    (target/'FINAL-RETRIEVAL-REPORT.md').write_text('\n'.join(lines)+'\n')
    print(str(target/'FINAL-RETRIEVAL-REPORT.md'))


if __name__=='__main__':main()
