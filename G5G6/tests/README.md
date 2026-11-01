# Offline checks

From `G5G6/`:

```bash
python tests/test_tiny.py
python tests/test_results.py
```

The6 numerical gates cover direct centered/uncentered eigenspace references, truncation error, zero/empty/few/repeated/NaN cases, G5 count allocation, G6 exhaustive small-knapsack reference and the single-token gradient identity. The5 publication checks recompute all44 run metrics from saved predictions, check sample visits/steps/budgets, exact holdout membership, pinned sources and five-/three-seed sample statistics. No network, GPU or pretrained weights are needed. Numerical testing needs CPU PyTorch.
