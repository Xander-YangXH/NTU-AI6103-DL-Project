# G5/G6 — Gradient-weighted EVA

The 2026-11-18 shared-settings rerun supersedes the original independent results. There are 18 tuning runs, 26 main runs and 6 low-resource SST-2 runs.

- G5 redistributes a total of 576 ranks. G6 reallocates ranks under a budget of 1,327,104 adapter parameters.
- RoBERTa-base, maximum length 128, training batch 32, BF16, 6% warm-up, zero weight decay and no gradient clipping.
- Select learning rates on the seed-2026 10% holdout with seed 0 and grid 2e-4, 5e-4, 1e-3. Main runs use the same 90% training split. Low-resource runs use 1,000 examples sampled with seed 2027 and reuse each group's full SST-2 learning rate.
- RTE/MRPC use 10 epochs and seeds 0–4. SST-2 uses 3 epochs and seeds 0–2. Low-resource SST-2 uses 20 epochs and seeds 0–2.
- Warm up the classification head for one epoch at 1e-3 with the encoder frozen. GW calibration preserves labels for the gradients, uses up to 1,024 training examples at batch 16, and caps rank at 16 (rho = 2). It uses a fixed sample budget.

See [results](results/summary.md), [selected learning rates](results/chosen_lr.json), and the [run table](results/g56_runs.csv). Each run has one JSON under results/main, results/tune or results/lowres. Official validation is evaluated after the final epoch, so per_epoch is empty.

From this directory, install dependencies in requirements.txt, then run:

```bash
export PYTHONPATH="$PWD/src"
python jobs/prefetch.py --help
python -m gw_eva.study
```

The original rerun used two processes on a single 4090 D. To reproduce that scheduling, run jobs/parallel_worker.py with distinct worker names and task sets rte mrpc and sst2 sst2_lowres in the same study directory.
