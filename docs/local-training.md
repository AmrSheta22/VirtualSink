# Run on this PC

Local execution is available at the user's explicit request. The repository's
HPC instructions still describe the separate Slurm workflow.

The virtual environment is `E:\repos\VirtualSink\venv`. Run its Python directly;
PowerShell activation and execution-policy changes are unnecessary.

Start all four variants sequentially with the default 3,000 steps per variant:

```powershell
Set-Location E:\repos\VirtualSink
.\venv\Scripts\python.exe -u run_ablation.py --device cuda --output-dir runs/local-trigger
```

The runner evaluates every 100 steps, saves the best validation checkpoint for
each variant, and may stop early on a plateau. It prints a comparison table and
saves the analysis plots after all four variants finish. Choose a new output
directory for every run; the runner refuses to overwrite existing results.

To force exactly 3,000 steps for every variant, add `--patience 0`. For CPU
execution, replace `--device cuda` with `--device cpu`.

Check the environment without starting training:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m pytest -q test_query_virtual_sink_attention.py test_trigger_experiment.py
```

## Controlled attention-only study

```powershell
.\venv\Scripts\python.exe -u run_controlled_ablation.py --device cuda --output-dir runs/attention-control
```

This runs 24 models: four attention types, two loss weightings, and three seeds.
Each model trains for exactly 3,000 steps. The architecture contains just one
attention layer with trainable Q/K/V/output projections and no residual, MLP,
normalization, or extra prediction head. The original dataset is unchanged.

`token_mean` gives equal weight to each token. `balanced` averages trigger MSE
and non-trigger MSE with weights 0.5/0.5; both losses use all input coordinates.
Metrics additionally separate Gaussian features (coordinates 3 onward) from
indicator coordinates. Checkpoints minimize the corresponding validation loss.

Training seeds 1729/1730/1731 are paired across the four arms and both losses.
A separate validation/test seed base (2718) holds evaluation examples fixed
across all runs and differs from the original experiment's evaluation data.
The runner saves source hashes, individual run logs/checkpoints/plots, a table
of per-seed results, paired query-minus-standard differences, and an aggregate
`REPORT.md` plus `controlled_comparison.png` with sample standard deviations.
These error bars describe variation across training seeds, not confidence bounds.

To run just one condition using the base runner:

```powershell
.\venv\Scripts\python.exe -u run_ablation.py --architecture attention_only --num-layers 1 --loss-mode balanced --device cuda --output-dir runs/attention-balanced-single
```

The Slurm-only training integration test is skipped locally. Recreate this
environment if needed with Python 3.10:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-local.txt
```
