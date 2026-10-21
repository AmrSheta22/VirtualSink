"""Produce static scientific figures from downloaded evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "artifacts/smollm-seed-analysis-20261016"
rows = json.loads((ROOT / "analysis.json").read_text())
curves = json.loads((ROOT / "curves.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(2,3,figsize=(13,7.1))
for ax, row in zip(axes.flat, rows):
    seed=row["seed"]
    for arm,color,label in (("standard","#52677e","Standard"),("query_sink","#008574","Query sink")):
        points=[c for c in curves if c["seed"]==seed and c["arm"]==arm]
        ax.plot([p["step"] for p in points],[p["validation_ce"] for p in points],color=color,lw=2,label=label)
    ax.set_title(f"Seed {seed}: test CE gain {row['ce_reduction']:+.3f}",fontweight="bold")
    ax.set_xlim((8,30) if seed<11 else (10,70))
    ax.set_ylim((7.2,8.1) if seed<11 else (6.5,8.05))
    ax.grid(alpha=.2)
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Validation cross-entropy")
axes[0,0].legend(frameon=False)
fig.suptitle("The sink changes the learning trajectory; the benefit depends on initialization",fontsize=15,fontweight="bold")
fig.text(.5,.01,"Top: 30-update / 1-step warmup recipe. Bottom: 70-update / 10-step warmup recipe. Early high-loss transients omitted from axes.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
fig.savefig(ROOT/"learning-curves.png",dpi=170)
fig.savefig(ROOT/"learning-curves.svg")
plt.close(fig)
fig,ax=plt.subplots(figsize=(12,3.8))
values=np.array([r["arms"]["query_sink"]["sink_mass_by_layer"] for r in rows])*100
im=ax.imshow(values,aspect="auto",cmap="YlGnBu",vmin=0,vmax=75)
ax.set_yticks(range(6),[f"Seed {r['seed']}  ({r['ce_reduction']:+.3f} CE gain)" for r in rows])
ax.set_xticks(np.arange(0,30,2),np.arange(1,31,2))
ax.set_xlabel("Layer (1-based)")
ax.set_title("Final sink mass has no single winning pattern",fontweight="bold")
fig.colorbar(im,ax=ax,label="Mean virtual mass (%), all test positions")
fig.tight_layout()
fig.savefig(ROOT/"sink-layer-mass.png",dpi=170)
plt.close(fig)
print("Figures saved")
