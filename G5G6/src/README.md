# Core modules

Install from `G5G6/` with `pip install -e . --no-deps`, or set `PYTHONPATH=src`.

| Module | Purpose |
|---|---|
| `gw_eva.spectral` | Uncentered/centered bounded-memory merge SVD, subspace geometry |
| `gw_eva.allocation` | G5 count budget; G6 exact integer prefix knapsack |
| `gw_eva.calibrate` | Mask-aligned72-layer forward/output-gradient hooks; frozen model |
| `gw_eva.runtime` | Model, target layers, data loader, head-only warm-up |
| `gw_eva.prepare` | Pinned public model download and checksums |
| `gw_eva.smoke` | Rank-varying PEFT injection, zero-B, actual update/reload checks |
| `gw_eva.study` | The recorded3-task tuning+main pipeline; full-epoch resume |
| `gw_eva.train` | Explicit short RTE-only training entry; not the26-run dispatcher |
| `gw_eva.hook_check` | Real72-layer frozen-graph/batch-scale/cleanup check |
| `gw_eva.tests` | Offline numerical gates |

Paths default to the enclosing `G5G6/`, independent of the original workstation. Model/data binaries and generated runtime outputs are ignored by git. Optional `GWEVA_ROOT` must point to a complete project root including `configs/`.
