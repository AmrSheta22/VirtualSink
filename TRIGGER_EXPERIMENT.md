# Trigger-conditional attention ablation

The four variants share a two-layer, pre-norm residual Transformer (64 hidden
features, four heads, 16 features per head, 128-wide GELU MLP). Only their attention
denominators differ. All shared weights start identically; all variants see the
same examples in the same order. Static and query sinks start at bias -3.5; query
sink weights use Normal(0, 1e-4). Quiet softmax uses a fixed sink logit of zero.

The on-demand dataset has one trigger at j in [2, 62]. Its target is the full
vector mean of inputs 1 through j-1, including coordinate 2. All other targets,
including BOS, are zero. Gaussian features remain present at BOS and the trigger.
Training, validation, and test have distinct deterministic seed streams.

## Execution

Dependencies: Python >=3.9, PyTorch >=1.12, matplotlib, numpy, pytest. The Phase 1
test suite additionally uses modern PyTorch SDPA and torch.func APIs.

This repository runs training only in Slurm. Install missing dependencies in the
HPC environment, commit/push the scripts, then use the standard helper:

```powershell
.\hpc\submit_hpc.ps1 -Script train_trigger_ablation.sh
```

The CPU job avoids the unresolved GPU device issue. It runs an end-to-end smoke
test and then trains all four models sequentially. Default training is 3,000
steps per variant, batch size 64, AdamW at 1e-3 with weight decay 1e-4, and gradient
norm clipping at 1. Validation occurs every 100 steps on 512 fixed sequences.
Early stopping requires ten evaluations without an improvement of at least 1e-7
and at least 1,000 steps. The best checkpoint is always the lowest observed
validation MSE, independently of the early-stopping tolerance. Test MSE is computed
only after loading that checkpoint, on 512 additional sequences. CPU runtime and
queue time depend on the allocated node; there is no fixed runtime guarantee.

The CLI can be customized inside an allocation:

```bash
python -u run_ablation.py --device cpu --output-dir runs/custom --patience 0
```

Outputs include config.json, per-variant best.pt and metrics.jsonl, summary.json,
comparison.csv, trigger_experiment_results.png/PDF, convergence.png/PDF,
mechanism_summary.json, and mechanism_example.npz. Run directories must be fresh
to prevent accidental checkpoint overwrites. The stored model config and CPU
state dictionaries make checkpoints portable; they are inference checkpoints,
not optimizer-resume snapshots.

Replot downloaded results locally without training:

```bash
python visualize_mechanisms.py --run-dir runs/trigger-JOBID --layer 0 --head 0
```

## Interpretation

Overall MSE assigns 63 of 64 positions to no-op behavior. Compare trigger MSE
against the zero-predictor baseline before interpreting low total loss as success.
Non-trigger MSE includes BOS; non-trigger *mechanism* summaries exclude BOS,
which has no earlier physical token. Metrics are weighted by token counts.

The four heatmaps use the same linear 0–1 color scale and a fixed held-out sample,
not a selected success case. The trajectory plots the observed sink logit alongside
the physical log-partition: their difference, rather than the sign of the sink
logit alone, determines offloading probability. Statistics are saved for every
layer and head across the test set. The selected head's hypothesis checks have
explicit descriptive thresholds and may legitimately fail.

Ordinary softmax enforces sum-to-one, but this architecture does not force BOS
usage: its value projections, residual paths, and MLP can implement alternatives.
Likewise, no constraint guarantees a negative trigger sink logit. A single-seed
toy comparison tests a mechanistic hypothesis; it does not prove theoretical
guarantees or eliminate alternative explanations. The dataset follows the supplied
equations; no unprovided paper-specific architectural constraints are assumed.

## Tests

```bash
python -m pytest -q test_query_virtual_sink_attention.py test_trigger_experiment.py
```

The end-to-end training/plotting test is skipped outside Slurm. Other tests check
exact targets, deterministic generation, matched initialization, all denominator
formulas, causality, gradients, and metric weighting.
