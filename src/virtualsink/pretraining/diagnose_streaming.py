"""Slurm GPU diagnostics: cache parity, eviction curves and per-head sink usage.

Uses two disjoint, deterministic 32769-token slices of the existing held-out
FineWeb test stream. These are packed documents, NOT claimed to be single books.
First 8192 predictions are within training length; subsequent ones test extrapolation.
Each completed policy/stream is saved atomically; requeues skip completed units.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .recipe import initialize, MODEL_CONFIG
from .streaming import StreamingDecoder
from .compare import compare_runs
from ..uptraining.native import set_window
from ..uptraining.qwen import disable_sinks


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


@torch.no_grad()
def parity(model, tokens, window, tolerance=0.003):
    set_window(model, window)
    hidden = model.model(tokens[:-1][None]).last_hidden_state[0]
    decoder = StreamingDecoder(model, window)
    maximum = 0.
    for i, token in enumerate(tokens[:-1]):
        logits, _, _, _ = decoder.step(token)
        reference = model.lm_head(hidden[i])
        maximum = max(maximum, (logits - reference).abs().max().item())
    if maximum > tolerance:
        raise ValueError(f"cache/parallel parity failed: {maximum} > {tolerance}")
    return dict(tokens=len(tokens)-1, max_absolute_logit_error=maximum, tolerance=tolerance)


def heatmap(path, values, title):
    rows, cols = len(values), len(values[0])
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="{70+rows*20}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="10" y="20" font-size="14">{title} (0=white, 1=blue)</text>']
    for l, row in enumerate(values):
        pieces.append(f'<text x="5" y="{56+l*20}" font-size="12">L{l}</text>')
        for h, v in enumerate(row):
            c = int(255 * (1-min(1., max(0., v))))
            pieces.append(f'<rect x="{45+h*55}" y="{40+l*20}" width="54" height="19" fill="rgb({c},{c},255)"><title>layer {l}, head {h}: {v:.6f}</title></rect>')
    path.write_text("\n".join(pieces + ["</svg>"]))


@torch.no_grad()
def evaluate_stream(model, tokens, capacity, prefix, window, tokenizer=None, training_length=8192):
    decoder = StreamingDecoder(model, capacity, prefix)
    losses, alpha, sinks, physical = [], [], [], []
    # Transfer diagnostics once per small block, avoiding per-head GPU syncs.
    for i, token in enumerate(tokens[:-1]):
        logits, a, s, p = decoder.step(token)
        ce = F.cross_entropy(logits[None], tokens[i+1:i+2]).item()
        if not math.isfinite(ce):
            raise FloatingPointError(f"nonfinite CE at input position {i}; inspect numerical health")
        losses.append(ce)
        alpha.append(a.cpu().numpy()); sinks.append(s.cpu().numpy()); physical.append(p.cpu().numpy())
        if (i+1) % 512 == 0:
            print(json.dumps(dict(event="stream_progress", positions=i+1, capacity=capacity, prefix=prefix, ce=float(np.mean(losses[-512:])))), flush=True)
    a, s, p = np.array(alpha), np.array(sinks), np.array(physical)
    bins = []
    for left in range(0, len(losses), window):
        ce = float(np.mean(losses[left:left+window]))
        bins.append(dict(input_position_start=left, input_position_end=min(left+window,len(losses)),
                         ce=ce, perplexity=math.exp(min(ce,700)),
                         extrapolation=left >= training_length,
                         mean_virtual_mass=float(a[left:left+window].mean())))
    mean = a.mean(0)
    semantic = []
    # One high/low example per head; observational query-token associations only.
    for layer in range(a.shape[1]):
        for head in range(a.shape[2]):
            for kind, pos in (("high",int(a[:,layer,head].argmax())), ("low",int(a[:,layer,head].argmin()))):
                tid = int(tokens[pos].item())
                semantic.append(dict(layer=layer, head=head, kind=kind, input_position=pos,
                    token_id=tid, token=tokenizer.decode([tid]) if tokenizer else None,
                    context=tokenizer.decode(tokens[max(0,pos-8):pos+1].tolist()) if tokenizer else None,
                    probability=float(a[pos,layer,head])))
    return dict(ce=float(np.mean(losses)), bins=bins, per_token_ce=losses, training_length=training_length,
        mean_virtual_mass_by_layer_head=mean.tolist(),
        max_virtual_mass_by_layer_head=a.max(0).tolist(),
        mean_sink_logit_by_layer_head=s.mean(0).tolist(),
        mean_physical_lse_by_layer_head=p.mean(0).tolist(),
        semantic_examples=semantic, max_cache_entries=max(k.shape[2] for k,v in decoder.cache),
        capacity=capacity, retain_prefix=prefix)


def write_report(out):
    files = sorted(out.glob("result-*.json"))
    lines = ["# Streaming eviction diagnostics", "",
             "Teacher-forced held-out packed FineWeb streams. Input positions are zero-based; FIFO first evicts at input position 512. Training length is recorded per result; positions beyond it test extrapolation.", "",
             "Loss increases are measured against the same model's full-cache reference on identical target tokens. Neither flat perplexity nor baseline collapse is assumed.", "",
             "| Variant | Stream | Policy | CE within training length | CE beyond training length | Mean excess CE vs full cache |", "|---|---:|---|---:|---:|---:|"]
    values = {f.stem[len("result-"):]:json.loads(f.read_text()) for f in files}
    comparisons=[]
    for name, result in values.items():
        variant, stream, policy = name.split("--")
        ce = np.array(result["per_token_ce"])
        ref = values.get(f"{variant}--{stream}--full")
        excess = float((ce - np.array(ref["per_token_ce"])).mean()) if ref else None
        trained = result.get("training_length",8192)
        beyond = float(ce[trained:].mean()) if len(ce)>trained else None
        lines.append(f"| {variant} | {stream} | {policy} | {ce[:trained].mean():.6f} | {beyond if beyond is not None else 'N/A'} | {excess if excess is not None else 'N/A'} |")
        if ref:
            rb = ref["bins"]
            deltas=[dict(position=b["input_position_start"],excess_ce=b["ce"]-r["ce"],ppl_ratio=math.exp(min(700,b["ce"]-r["ce"]))) for b,r in zip(result["bins"],rb)]
            comparisons.append(dict(variant=variant,stream=stream,policy=policy,bins=deltas))
    (out/"REPORT.md").write_text("\n".join(lines))
    atomic_json(out/"eviction_comparison.json",comparisons)
    if values:
        ymax=max(b["ce"] for v in values.values() for b in v["bins"])*1.05
        xmax=max(b["input_position_end"] for v in values.values() for b in v["bins"])
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="40" y="20">Cross-entropy by input position (512-token bins)</text>']
        colors=["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2","#374151"]
        for j,(name,result) in enumerate(values.items()):
            color=colors[j%len(colors)]
            pts=" ".join(f'{40+900*b["input_position_end"]/xmax:.1f},{400-350*b["ce"]/ymax:.1f}' for b in result["bins"])
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/>')
            svg.append(f'<text x="{40+(j%2)*470}" y="{455+(j//2)*23}" fill="{color}" font-size="12">{name}</text>')
        for pos in (512,8192,32768):
            x=40+900*pos/xmax
            svg.append(f'<path d="M{x} 40 V400" stroke="gray" stroke-dasharray="4 4"/><text x="{x}" y="420">{pos}</text>')
        for fraction in (0,.25,.5,.75,1):
            svg.append(f'<text x="0" y="{400-350*fraction}">{ymax*fraction:.1f}</text>')
        (out/"loss_by_position.svg").write_text("\n".join(svg+["</svg>"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--window",type=int,default=512)
    parser.add_argument("--length",type=int,default=32768)
    parser.add_argument("--streams",type=int,default=2)
    args=parser.parse_args()
    if args.window<=4 or args.streams<1 or args.length<8192:
        parser.error("window >4, streams >=1, length >=8192 required")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        parser.error("Slurm GPU allocation required")
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    # Reuse the strict finished-pair checks before evaluating either model.
    compare_runs(args.root/"runs/standard",args.root/"runs/query_sink")
    cfg=json.loads((args.root/"runs/standard/config.json").read_text())
    data=Path(cfg["data_dir"])
    manifest=json.loads((data/"manifest.json").read_text())
    if digest(data/"manifest.json") != cfg["manifest_sha256"]:
        raise ValueError("data manifest changed")
    entry=manifest["sources"]["fineweb"]["splits"]["test"]
    path=data/entry["file"]
    if digest(path)!=entry["sha256"]: raise ValueError("test token checksum mismatch")
    stream=np.memmap(path,dtype="<u4",mode="r")
    if len(stream)<args.streams*(args.length+1): raise ValueError("insufficient disjoint test tokens")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(hf_hub_download(manifest["tokenizer"],"tokenizer.json",revision=manifest["tokenizer_revision"],local_files_only=True))
    run_identity=dict(window=args.window,length=args.length,streams=args.streams,
        manifest_sha256=cfg["manifest_sha256"],source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        checkpoint_sha256={v:digest(args.root/"runs"/v/"final.pt") for v in ("standard","query_sink")},
        source="disjoint consecutive slices of packed held-out FineWeb; not single documents")
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=run_identity:
        raise ValueError("diagnostic resume configuration mismatch")
    atomic_json(out/"config.json",run_identity)
    started=time.monotonic()
    for variant in ("standard","query_sink"):
        checkpoint=torch.load(args.root/"runs"/variant/"final.pt",map_location="cpu")
        c=checkpoint["config"]
        model=initialize(c["model_seed"],32,dict(MODEL_CONFIG,max_position_embeddings=c["sequence_length"]),sink_weight_std=c["sink_weight_std"],sink_bias=c["sink_bias"])
        initial={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith(("sink_w","sink_b"))}
        model.load_state_dict(checkpoint["model"]); del checkpoint
        params={}
        for n,p in model.named_parameters():
            if n in initial:
                params[n]=dict(initial=initial[n].tolist(),trained=p.detach().tolist(),
                    initial_norm=float(initial[n].norm()),trained_norm=float(p.detach().norm()),delta_norm=float((p.detach()-initial[n]).norm()))
                if n.endswith("sink_w"):
                    params[n].update(initial_norm_by_head=initial[n].norm(dim=-1).tolist(),
                                     trained_norm_by_head=p.detach().norm(dim=-1).tolist())
        atomic_json(out/f"parameters-{variant}.json",params)
        model.cuda().eval(); disable_sinks(model,variant=="standard")
        if not (out/f"parity-{variant}.json").exists():
            tokens=torch.tensor(np.asarray(stream[:8193],dtype=np.int64),device="cuda")
            checks={"fifo":parity(model,tokens,args.window),"full":parity(model,tokens,None)}
            atomic_json(out/f"parity-{variant}.json",checks)
        policies=[("full",None,0,variant=="standard"),("fifo",args.window,0,variant=="standard"),("prefix4",args.window,4,variant=="standard")]
        if variant=="query_sink": policies.append(("fifo_disabled",args.window,0,True))
        for sid in range(args.streams):
            start=sid*(args.length+1)
            tokens=torch.tensor(np.asarray(stream[start:start+args.length+1],dtype=np.int64),device="cuda")
            for policy,cap,prefix,disabled in policies:
                dest=out/f"result-{variant}--{sid}--{policy}.json"
                if dest.exists(): continue
                if time.monotonic()-started>18*3600: raise SystemExit(75)
                disable_sinks(model,disabled)
                result=evaluate_stream(model,tokens,cap,prefix,args.window,tok,training_length=c["sequence_length"])
                atomic_json(dest,result)
                heatmap(dest.with_suffix(".svg"),result["mean_virtual_mass_by_layer_head"],f"{variant} {policy} stream {sid}")
                write_report(out)
        del model
        torch.cuda.empty_cache()
    write_report(out)
    print("STREAMING DIAGNOSTICS COMPLETE",flush=True)


if __name__=="__main__": main()
