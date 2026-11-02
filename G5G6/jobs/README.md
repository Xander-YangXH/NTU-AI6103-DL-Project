# Jobs

Run after installing the package from `G5G6/`.

- `prefetch.py [--tasks rte mrpc sst2]`: model and data revisions pinned by `configs/`; verifies public download hashes. `GW_HF_ENDPOINT` may optionally select a mirror; hashes/revisions remain fixed.
- `run_queue.sh --tasks rte`: recorded independent protocol, head warm-up → fit-only calibration → LR selection → full-train seed pairs. Use a fresh runtime root for a new experiment. Existing complete jobs are skipped.
- `g56.sbatch`: optional Slurm template, **not executed on TC2**. Set `GWEVA_CODE`, `TASK` and site-specific account/QoS/partition. One task perjob; the full original queue exceeded6h.
- `summarize.py`: read-only summary from exported run JSONs; no GPU/data needed.
- `plot_diagnostics.py`: derive rank heatmaps and first500-loss figures from the exported measurements.

The full26-run historical queue took about10.02h including tuning/preparation, on one4090. These timings are not predictions for a different cluster.
