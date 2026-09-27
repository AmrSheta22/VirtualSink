"""Read-only final-checkpoint interventions on saturated SWA virtual sinks."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from virtualsink.pretraining.recipe import initialize, MODEL_CONFIG
from virtualsink.pretraining.data import MixedBlocks
from virtualsink.uptraining import native
from virtualsink.uptraining.data import sha256


@torch.no_grad()
def evaluate_block(model, tokens, disabled_heads, observed_heads, disable_all=False):
    """Disable only selected virtual logits, retaining physical keys and values.

Measure each observed head's raw output L2 norm and its individual contribution
through the corresponding columns of o_proj, before cross-head summation.
"""
    model.eval()
    original = native.sink_attention
    calls = 0
    stats = {}
    def attention(q, k, v, w, b, chunk_size=64, recompute=False, window_size=None):
        nonlocal calls
        layer = calls
        calls += 1
        affected = [h for l,h in disabled_heads if l==layer]
        if disable_all: affected = list(range(q.shape[1]))
        if affected:
            w,b = w.clone(),b.clone()
            w[affected] = 0.; b[affected] = -1e6
        out,gates=original(q,k,v,w,b,chunk_size,False,window_size)
        attn=model.model.layers[layer].self_attn
        for l,h in observed_heads:
            if l!=layer: continue
            raw=out[0,h].float()
            matrix=attn.o_proj.weight[:,h*attn.head_dim:(h+1)*attn.head_dim]
            projected=[]
            for start in range(0,len(raw),128):
                projected.append(F.linear(raw[start:start+128],matrix).norm(dim=-1))
            stats[f"L{l}H{h}"]=dict(raw_output_norm=float(raw.norm(dim=-1).mean()),
                projected_output_norm=float(torch.cat(projected).mean()),
                physical_mass=float(gates[0,h].mean()))
        return out,gates
    native.sink_attention=attention
    try:
        hidden=model.model(tokens[:-1][None]).last_hidden_state[0]
        losses=[]
        for start in range(0,len(hidden),64):
            logits=model.lm_head(hidden[start:start+64]).float()
            losses.append(F.cross_entropy(logits,tokens[start+1:start+1+len(logits)],reduction="none").cpu())
        losses=torch.cat(losses).numpy()
        if not np.isfinite(losses).all(): raise FloatingPointError("nonfinite token loss")
        return losses,stats
    finally:
        native.sink_attention=original


def atomic_json(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("Slurm GPU allocation required")
    run=args.root/"runs/query_sink"
    if json.loads((run/"status.json").read_text()).get("status")!="complete":
        raise ValueError("training must be complete")
    checkpoint=torch.load(run/"final.pt",map_location="cpu")
    cfg=checkpoint["config"]
    for name,digest in cfg["source_sha256"].items():
        if sha256(args.root/"release/src/virtualsink"/name)!=digest:
            raise ValueError(f"training source changed: {name}")
    if sha256(Path(cfg["data_dir"])/"manifest.json")!=cfg["manifest_sha256"]:
        raise ValueError("data manifest changed")
    summary=json.loads((run/"summary.json").read_text())
    heads=[(l,h) for l,row in enumerate(summary["attention"]["by_layer_head"])
           for h,values in enumerate(row) if values[2]>=.99]
    if not heads: raise ValueError("no saturated heads")
    model=initialize(cfg["model_seed"],32,dict(MODEL_CONFIG,max_position_embeddings=cfg["sequence_length"]),
                     sink_weight_std=cfg["sink_weight_std"],sink_bias=cfg["sink_bias"])
    model.load_state_dict(checkpoint["model"]); del checkpoint
    native.set_window(model,cfg["window_size"] or None)
    model.cuda().eval()
    data=MixedBlocks(Path(cfg["data_dir"]),"test")
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    manifest=json.loads((Path(cfg["data_dir"])/"manifest.json").read_text())
    tokenizer=Tokenizer.from_file(hf_hub_download(manifest["tokenizer"],"tokenizer.json",revision=manifest["tokenizer_revision"],local_files_only=True))
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    identity=dict(checkpoint_sha256=sha256(run/"final.pt"),script_sha256=sha256(Path(__file__)),
                  manifest_sha256=cfg["manifest_sha256"],heads=heads,blocks=len(data),sequence_length=data.length)
    identity=json.loads(json.dumps(identity))
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=identity:
        raise ValueError("intervention resume mismatch")
    atomic_json(out/"config.json",identity)
    conditions=[("enabled",[],False)]+[(f"L{l}H{h}",[(l,h)],False) for l,h in heads]+[("all_saturated",heads,False),("all_sinks",[],True)]
    for name,disabled,all_off in conditions:
        for block in range(len(data)):
            dest=out/f"{name}-{block:03}.npz"
            if dest.exists(): continue
            losses,stats=evaluate_block(model,data[block].cuda(),disabled,heads,all_off)
            temp=dest.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.savez_compressed(f,losses=losses,stats=json.dumps(stats))
            temp.replace(dest)
            print(json.dumps(dict(condition=name,block=block,ce=float(losses.mean()))),flush=True)
    baseline=np.stack([np.load(out/f"enabled-{i:03}.npz")["losses"] for i in range(len(data))])
    reports=[]
    for name,_,_ in conditions:
        losses=[]; observed=[]
        for i in range(len(data)):
            with np.load(out/f"{name}-{i:03}.npz") as saved:
                losses.append(saved["losses"]); observed.append(json.loads(str(saved["stats"])))
        delta=np.stack(losses)-baseline
        block_means=delta.mean(1)
        flat=delta.flatten()
        indices=np.argsort(flat)
        examples=[]
        for kind,selected in (("most_harmed",indices[-20:][::-1]),("most_helped",indices[:20])):
            for index in selected:
                block,pos=divmod(int(index),data.length)
                tokens=data[block]
                examples.append(dict(kind=kind,block=block,input_position=pos,target_id=int(tokens[pos+1]),
                    target=tokenizer.decode([int(tokens[pos+1])]),
                    preceding_text=tokenizer.decode(tokens[max(0,pos-24):pos+1].tolist()),delta_ce=float(flat[index])))
        norms={key:{metric:float(np.mean([s[key][metric] for s in observed])) for metric in observed[0][key]} for key in observed[0]}
        rng=np.random.RandomState(1729)
        bootstrap=np.array([rng.choice(block_means,len(block_means),replace=True).mean() for _ in range(2000)])
        report=dict(condition=name,mean_delta_ce=float(flat.mean()),
            block_bootstrap_95pct_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
            quantiles=dict(zip(["min","p01","p50","p95","p99","p999","max"],np.quantile(flat,[0,.01,.5,.95,.99,.999,1]).tolist())),
            harmed_over_0_1_fraction=float((flat>.1).mean()),harmed_over_1_fraction=float((flat>1).mean()),
            helped_over_0_1_fraction=float((flat<-.1).mean()),
            head_output_statistics=norms,examples=examples)
        reports.append(report)
    atomic_json(out/"summary.json",reports)
    lines=["# Individual SWA sink interventions","","Positive delta CE means disabling worsens prediction. Token tails and projected head-output norms are reported separately from average loss.","",
           "| Disabled sinks | Mean delta CE | p99 token delta | Max token delta | Tokens harmed >0.1 CE |","|---|---:|---:|---:|---:|"]
    for r in reports:
        lines.append(f'| {r["condition"]} | {r["mean_delta_ce"]:.8f} | {r["quantiles"]["p99"]:.6f} | {r["quantiles"]["max"]:.6f} | {100*r["harmed_over_0_1_fraction"]:.4f}% |')
    (out/"REPORT.md").write_text("\n".join(lines))
    print("HEAD INTERVENTIONS COMPLETE",flush=True)


if __name__=="__main__": main()
