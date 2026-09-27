"""Rebuild Qwen-only report figures from collected experiment JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return [json.loads(x) for x in (ROOT/path/'metrics.jsonl').read_text().splitlines()]
series={
 'Standard SWA':rows(Path('runs/swa-1024-512/runs/qwen05-standard')),
 'Virtual sinks + penalty':rows(Path('runs/swa-1024-512/runs/qwen05-query_sink')),
 'Virtual sinks, no penalty (interrupted)':rows(Path('runs/swa-noaux-1024-512/runs/qwen05-query_sink'))}
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for label,rs in series.items():
 tr=[r for r in rs if 'lm_ce' in r]
 va=[r for r in rs if 'validation' in r]
 x=np.array([r['step'] for r in tr]);y=np.array([r['lm_ce'] for r in tr])
 ax[0].plot(x[24:],np.convolve(y,np.ones(25)/25,mode='valid'),label=label)
 ax[1].plot([r['step'] for r in va],[r['validation']['perplexity'] for r in va],label=label)
ax[0].set_title('Training CE: 25-step moving average');ax[0].set_ylabel('Cross-entropy (nats/token)')
ax[1].set_title('Validation perplexity: same 32 packed blocks');ax[1].set_ylabel('Perplexity');ax[1].set_ylim(10.2,12.5)
for a in ax:a.set_xlabel('Optimizer step');a.grid(alpha=.2)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(OUT/'training.png',dpi=170);plt.close(fig)

data=read('runs/streaming-confirmation-20260913/article_results.json')
data=[r for r in data if r['model']=='qwen05']
names=[];delta=[]
for i in range(12):
 st=next(r for r in data if r['article_index']==i and r['policy']=='standard-fifo')
 si=next(r for r in data if r['article_index']==i and r['policy']=='query_sink-fifo')
 names.append(st['title'].strip('= ').replace('@-@','-'));delta.append(st['post_eviction']['ce']-si['post_eviction']['ce'])
fig,ax=plt.subplots(figsize=(10,6))
ax.barh(names,delta,color='#2563eb');ax.invert_yaxis();ax.set_xlim(0,.21)
ax.set_xlabel('Standard FIFO CE minus virtual-sink FIFO CE (nats/token)')
ax.set_title('Qwen: virtual sinks improve post-eviction loss on all 12 new articles')
ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT/'confirmation.png',dpi=170);plt.close(fig)

p=Path('runs/uptrained-streaming-20260913/summary/qwen05')
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
policies=[('standard-fifo','Standard FIFO','#2563eb'),('standard-prefix4','Standard prefix4, absolute positions','#16a34a'),('query_sink-fifo','Virtual-sink FIFO','#dc2626'),('query_sink-disabled','Sink checkpoint, sinks disabled','#9333ea')]
for i,ax in enumerate(axes):
 for policy,label,color in policies:
  losses=np.array(read(p/f'{policy}-article{i}.json')['per_token_ce'])
  ax.plot(np.arange(256,8192,512),[losses[j:j+512].mean() for j in range(0,8192,512)],label=label,color=color)
 ax.axvline(512,color='black',linestyle='--',alpha=.5);ax.axvline(1024,color='gray',linestyle=':',alpha=.5)
 ax.set_title(['American Beauty','2003 Pacific typhoon season'][i]);ax.set_xlabel('Input position (zero-based)');ax.set_ylabel('CE in 512-token bins');ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8)
fig.suptitle('Qwen exploratory streaming: dashed = eviction; dotted = uptraining length')
fig.tight_layout(rect=[0,.11,1,.93]);fig.savefig(OUT/'streaming.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,6))
for i,period in enumerate(['before','after']):
 mass=np.mean([read(p/f'query_sink-fifo-article{j}.json')[period+'_mean_virtual_by_layer_head'] for j in range(2)],axis=0)
 im=axes[i].imshow(mass,vmin=0,vmax=1,aspect='auto',cmap='Blues')
 axes[i].set_title(f'{period.capitalize()} eviction: {100*mass.mean():.2f}% mean');axes[i].set_xlabel('Head (zero-based)');axes[i].set_ylabel('Layer (zero-based)')
fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.04,label='Virtual-sink probability')
fig.savefig(OUT/'heads.png',dpi=170,bbox_inches='tight');plt.close(fig)
