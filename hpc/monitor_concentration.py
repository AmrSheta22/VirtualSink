"""Read-only concentration monitor of existing per-head validation diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def summarize(attention):
    column=attention["columns"].index("virtual_mass")
    heads=sorted([(float(h[column]),l,i) for l,row in enumerate(attention["by_layer_head"]) for i,h in enumerate(row)],reverse=True)
    values=[v for v,l,h in heads]; total=sum(values)
    return dict(mean=total/len(values),minimum=min(values),maximum=max(values),
        heads=len(values),over_30pct=sum(v>=.3 for v in values),over_99pct=sum(v>=.99 for v in values),
        below_0_1pct=sum(v<.001 for v in values),
        top7_share=sum(values[:7])/total if total else None,
        effective_heads=total**2/sum(v*v for v in values) if total else None,
        top_heads=[dict(layer=l,head=h,probability=v) for v,l,h in heads[:20]])


def collect(root,reference,out):
    old=summarize(json.loads(reference.read_text())["attention"])
    result=dict(reference_swa=old,runs={})
    lines=["# Virtual-sink concentration","",
        "Probabilities are per-head means on each run's sampled late validation queries. Sequence lengths, attention masks and samples differ from the old SWA final test; compare concentration patterns, not absolute probabilities as a controlled comparison.","",
        "Effective heads = (sum probabilities)^2 / sum squared probabilities; 270 means uniform allocation, 7 means approximately seven equally active heads.","",
        "| Run / update | Mean sink mass | Maximum head | Heads >=30% | Heads >=99% | Share in top 7 | Effective heads |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def row(label,s):
        share=f'{100*s["top7_share"]:.2f}%' if s["top7_share"] is not None else "N/A"
        effective=f'{s["effective_heads"]:.2f}' if s["effective_heads"] is not None else "N/A"
        return f'| {label} | {100*s["mean"]:.3f}% | {100*s["maximum"]:.3f}% | {s["over_30pct"]} | {s["over_99pct"]} | {share} | {effective} |'
    lines.append(row("Old SWA final",old))
    for variant in ("standard","query_sink"):
        path=root/"runs"/variant/"metrics.jsonl"
        by_step={}
        if path.exists():
            for text in path.read_text().splitlines():
                try: record=json.loads(text)
                except ValueError: continue  # Concurrent writer may leave a partial final line.
                if "attention" not in record: continue
                by_step[record["step"]]=dict(step=record["step"],utc=record.get("utc"),**summarize(record["attention"]))
        history=[by_step[k] for k in sorted(by_step)]
        result["runs"][variant]=history
        for s in history: lines.append(row(f'{variant} / {s["step"]}',s))
    for variant,history in result["runs"].items():
        if history:
            lines.extend(["",f"Latest {variant} top heads (zero-based layer/head):",""])
            for h in history[-1]["top_heads"][:10]:
                lines.append(f'- L{h["layer"]} H{h["head"]}: {100*h["probability"]:.3f}%')
    out.mkdir(parents=True,exist_ok=True)
    for name,contents in (("latest.json",json.dumps(result,indent=2)),("REPORT.md","\n".join(lines))):
        dest=out/name; temp=dest.with_suffix(".tmp"); temp.write_text(contents); temp.replace(dest)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--watch",type=int,default=0)
    p.add_argument("--jobs",default="214473,214474")
    args=p.parse_args(); started=time.monotonic(); previous=None
    while True:
        result=collect(args.root,args.reference,args.output)
        current=json.dumps(result,sort_keys=True)
        if current!=previous:
            print(json.dumps({v:rows[-1:] for v,rows in result["runs"].items()}),flush=True)
            previous=current
        if not args.watch: return
        if all((args.root/"runs"/v/"summary.json").exists() for v in ("standard","query_sink")): return
        status=subprocess.run(["sacct","-X","-n","-P","-j",args.jobs,"--format=JobID,State"],capture_output=True,text=True,timeout=20)
        states={r.split("|")[0]:r.split("|")[1] for r in status.stdout.splitlines() if "|" in r}
        terminal=("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL")
        if all(any(states.get(j,"").startswith(s) for s in terminal) for j in args.jobs.split(",")):
            print("Monitored jobs terminated: "+json.dumps(states),flush=True); return
        if time.monotonic()-started>23*3600: raise SystemExit(75)
        time.sleep(args.watch)


if __name__=="__main__": main()
