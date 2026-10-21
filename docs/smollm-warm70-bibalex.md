# SmolLM 135M, 70 updates with a 10-update warmup

Follow-up to the full-512 three-seed experiment: initialization-sensitive gains
motivated a smoother learning-rate ramp and a larger optimizer-update budget.
The user explicitly selected **70 updates**. This experiment uses new model
seeds **11, 12, 13**, each with matched standard and query-sink arms.

The linear warmup has scheduler multipliers 0, 0.1, ..., 1 over scheduler steps
0 through 10, reaching learning rate 0.003. The first optimizer update remains
zero-rate under the existing LambdaLR convention. Peak learning rate is first
used on optimizer update 11. The plateau/cooldown retains the shortened original
recipe, with cooldown beginning at scheduler step 58 of 70.

The architecture, sink weight standard deviation 0.02, sink bias 4, full causal
context 512, FP32, microbatch16, global batch2048, chunk32, AdamW, clipping and
no-auxiliary-loss objective match the previous experiment. The release is staged
from the frozen completed 20261009 source, with only the explicit-warmup
recipe/trainer changes and schedule regression test overlaid. Newer unrelated
attention changes in the shared workspace are not deployed. **21 targeted local
tests passed** against this staged release.

Training consumes 73,400,320 input tokens per arm. Data preparation extends the
same pinned source/tokenizer recipe for 70 updates and must verify that all
original validation/test token streams and order hashes remain identical.
Training starts only after data preparation and verification succeed.

## Jobs and outputs

Root: `/cluster/users/alex116u1/data/smollm-warm70-20261014`.

- CPU preparation: **217107**.
- GPU array: **217108**, tasks 0/1 seed11 standard/query_sink, tasks2/3 seed12,
  tasks4/5 seed13. Up to six arms run concurrently on K80 devices.
- Paired comparisons: **217109**, **217110**, **217111**.
- `jobs.tsv`, `protocol.json` and `source-sha256.json` record the submitted
  protocol and provenance. Original experiment outputs are preserved.

Jobs checkpoint partial gradient accumulation and requeue across 24-hour GPU
allocations, with a trainer pause after23 hours. Fixed final checkpoints are
evaluated on all2048 test blocks, including the trained sink-disabled intervention.
Paired comparisons enforce matching configurations and initialization hashes.

Run local monitoring with:

```powershell
.\venv\Scripts\python.exe hpc/report_smollm_warm70.py
```

It saves snapshots under `artifacts/smollm-warm70-20261014` and writes the combined
aggregate when all three comparisons finish. Three new seeds plus simultaneous
warmup/budget changes cannot isolate the causal effect of warmup alone.
