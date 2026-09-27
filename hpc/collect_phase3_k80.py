"""Read-only HPC collection and local reporting; does not submit or edit jobs."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", action="store_true", help="download completed-run best checkpoints and verify SHA256")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    local = repo / "runs" / "k80-pilots"
    local.mkdir(parents=True, exist_ok=True)
    host = "alex116u1@login02.c2.hpc.bibalex.org"
    remote = "/cluster/users/alex116u1/data/phase3-k80-20260909"
    executable = Path(os.environ["WINDIR"]) / "System32" / "OpenSSH"
    options = ["-i", str(repo / "server_access/private_key"), "-o", "IdentitiesOnly=yes",
               "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o",
               "UserKnownHostsFile=" + str(repo / "server_access/known_hosts"), "-o", "ConnectTimeout=20"]
    def ssh(command):
        return subprocess.run([str(executable / "ssh.exe"), "-T", *options, host, command],
                              check=True, capture_output=True, text=True).stdout
    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(executable / "scp.exe"), *options, host + ":" + source, str(destination)], check=True)
    ids = ssh("cat " + shlex.quote(remote + "/job_ids.txt")).split()
    if len(ids) != 3 or not all(i.isdigit() for i in ids):
        raise ValueError("unexpected job ID file")
    accounting = ssh("sacct -j " + ",".join(ids) + " --format=JobID,State,ExitCode,NodeList,Elapsed,Start,End --parsable2")
    updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (local / "accounting.txt").write_text(updated + "\n" + accounting, encoding="utf-8")
    (local / "job_ids.json").write_text(json.dumps(dict(zip(("prepare", "smollm135", "qwen05"), ids)), indent=2))
    files = ssh("find " + shlex.quote(remote) + " -maxdepth 4 -type f "
                r"\( -name '*.json' -o -name '*.jsonl' -o -name '*.out' -o -name '*.err' -o -name 'environment-freeze.txt' \)").splitlines()
    for source in files:
        relative = Path(source).relative_to(remote)
        if relative.parts[0] not in ("logs", "runs", "data", "environment-freeze.txt"):
            continue
        copy(source, local / relative)
    listing = ssh("find " + shlex.quote(remote + "/runs") + " -maxdepth 2 -name '*.pt' -printf '%p %s bytes\\n' 2>/dev/null || true")
    (local / "remote-checkpoints.txt").write_text(updated + "\n" + listing, encoding="utf-8")
    lines = ["# VirtualSink K80 run status", "", f"Collected at {updated}.", "",
             "```text", accounting.strip(), "```", "",
             "Both runs target 1,000 optimizer steps / 8,192,000 input tokens, sequence length 256, "
             "microbatch 1 and accumulation 32. SmolLM-135M trains the full backbone; Qwen2.5-0.5B trains "
             "attention projections and query sinks. Both use FP32 and gamma 0.01 anti-collapse loss.", ""]
    for slug in ("smollm135", "qwen05"):
        path = local / "runs" / slug
        lines += [f"## {slug}", ""]
        metrics = path / "metrics.jsonl"
        rows = []
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a concurrently appended last line may be incomplete
        training = [r for r in rows if "lm_ce" in r]
        if training:
            row = training[-1]
            lines += [f"Latest completed step: {row['step']}. Training CE: {row['lm_ce']:.6f}; "
                      f"auxiliary penalty: {row['auxiliary']:.6f}. "
                      f"Step throughput: {row['tokens_per_second']:.1f} input tokens/s. "
                      f"Peak allocated VRAM: {row['peak_allocated_bytes'] / 2**30:.2f} GiB.", ""]
        else:
            lines += ["No completed optimizer step observed yet.", ""]
        for label in ("pretrained_validation", "adapted_initial_validation", "validation"):
            matches = [r for r in rows if label in r]
            if matches:
                row = matches[-1]
                value = row[label]
                lines += [f"{label} at step {row['step']}: CE {value['ce']:.6f}, perplexity "
                          f"{value['perplexity']:.4f}; {value['heads_below_0.01']} heads have mean physical gate below 0.01.", ""]
        timing_path = path / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            low, high = timing["remaining_observed_range_seconds"]
            lines += [f"Measured estimate at step {timing['step']}: {timing['remaining_seconds']/3600:.2f} hours remaining "
                      f"(observed-step range {low/3600:.2f}–{high/3600:.2f} hours). "
                      "Includes estimated validation/checkpoint overhead; excludes queue and setup. This is not a confidence interval.", ""]
        summary = path / "summary.json"
        if summary.exists():
            lines += ["Final validation-selected checkpoint results:", "", "```json", summary.read_text(), "```", ""]
            if args.checkpoints:
                source = remote + "/runs/" + slug + "/best.pt"
                digest = ssh("sha256sum " + shlex.quote(source)).split()[0]
                destination = path / "best.pt"
                copy(source, destination)
                hasher = hashlib.sha256()
                with destination.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(block)
                if hasher.hexdigest() != digest:
                    raise ValueError("checkpoint download checksum mismatch")
                (path / "best.pt.sha256").write_text(digest + "\n")
                lines += [f"Best checkpoint downloaded and SHA256 verified: `{digest}`.", ""]
    lines += ["## Saved checkpoints on HPC", "", "```text", listing.strip() or "None observed yet.", "```", "",
              "Tokenizers differ, so perplexities across models are not directly comparable. "
              "These are two sink-training pilots, not matched trained standard-attention controls. "
              "The 24-hour allocation is a ceiling, not a measured runtime estimate.", ""]
    (local / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(local / "REPORT.md")


if __name__ == "__main__":
    main()
