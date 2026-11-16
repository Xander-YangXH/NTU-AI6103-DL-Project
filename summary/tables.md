## Shared protocol: official validation, mean ± sd (%)

| Task | G1 | G2 | G3 | G4 | G7 | G9 |
|---|---|---|---|---|---|---|
| SST-2 (acc) | 93.92 ± 0.00 | 94.07 ± 0.24 | 94.19 ± 0.43 | 93.65 ± 0.24 | 93.96 ± 0.46 | 94.00 ± 0.63 |
| MRPC (F1) | 91.34 ± 0.43 | 91.16 ± 1.11 | 92.02 ± 0.79 | 91.50 ± 0.68 | 91.83 ± 0.46 | 91.50 ± 0.60 |
| RTE (acc) | 73.94 ± 2.10 | 73.86 ± 1.04 | 74.80 ± 2.49 | 73.57 ± 2.06 | 56.68 ± 8.85 | 75.67 ± 1.70 |
| Low-res SST-2 1k (acc) | 91.44 ± 0.28 | – | 88.81 ± 0.83 | – | – | – |

Seeds: SST-2 3, MRPC / RTE / low-resource 5. Learning rates:

| Task | G1 | G2 | G3 | G4 | G7 | G9 |
|---|---|---|---|---|---|---|
| SST-2 (acc) | 0.0005 | 0.0005 | 0.001 | 0.0005 | 0.0002 | 0.001 |
| MRPC (F1) | 0.0002 | 0.001 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| RTE (acc) | 0.0005 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| Low-res SST-2 1k (acc) | 0.0005 | – | 0.001 | – | – | – |

## Paired difference vs G1 (same seeds, percentage points, 95% CI)

| Task | G2 − G1 | G3 − G1 | G4 − G1 | G7 − G1 | G9 − G1 |
|---|---|---|---|---|---|
| SST-2 (acc) | +0.15 [-0.44, +0.75] | +0.27 [-0.81, +1.35] | -0.27 [-0.86, +0.33] | +0.04 [-1.11, +1.19] | +0.08 [-1.49, +1.65] |
| MRPC (F1) | -0.17 [-1.59, +1.24] | +0.68 [-0.60, +1.97] | +0.16 [-1.21, +1.54] | +0.49 [-0.44, +1.41] | +0.17 [-0.82, +1.16] |
| RTE (acc) | -0.07 [-3.23, +3.09] | +0.87 [-3.27, +5.01] | -0.36 [-3.14, +2.42] | -17.26 [-27.93, -6.59] * | +1.73 [-2.67, +6.14] |
| Low-res SST-2 1k (acc) | – | -2.64 [-3.70, -1.57] * | – | – | – |

`*` the interval excludes 0.

## Paired difference vs G3 (same seeds, percentage points, 95% CI)

| Task | G1 − G3 | G2 − G3 | G4 − G3 | G7 − G3 | G9 − G3 |
|---|---|---|---|---|---|
| SST-2 (acc) | -0.27 [-1.35, +0.81] | -0.11 [-0.61, +0.38] | -0.54 [-1.69, +0.62] | -0.23 [-2.45, +2.00] | -0.19 [-1.63, +1.24] |
| MRPC (F1) | -0.68 [-1.97, +0.60] | -0.86 [-2.86, +1.14] | -0.52 [-1.56, +0.52] | -0.19 [-0.76, +0.37] | -0.52 [-1.21, +0.17] |
| RTE (acc) | -0.87 [-5.01, +3.27] | -0.94 [-3.10, +1.22] | -1.23 [-4.31, +1.85] | -18.12 [-28.87, -7.38] * | +0.87 [-3.31, +5.04] |
| Low-res SST-2 1k (acc) | +2.64 [+1.57, +3.70] * | – | – | – | – |

`*` the interval excludes 0.

## Step-0 ‖∂L/∂B‖ (mean over seeds)

| Task | G1 | G2 | G3 | G4 | G7 | G9 |
|---|---|---|---|---|---|---|
| SST-2 (acc) | 1.61 | 2.95 | 25.06 | 25.74 | 59.41 | 25.26 |
| MRPC (F1) | 0.24 | 0.42 | 1.63 | 1.62 | 7.67 | 2.75 |
| RTE (acc) | 0.16 | 0.29 | 1.35 | 1.33 | 2.47 | 1.41 |
| Low-res SST-2 1k (acc) | 0.18 | – | 2.19 | – | – | – |

## Environment per group

| Group | Runs | GPU | torch / transformers / peft | LoRA params |
|---|---|---|---|---|
| G1 | 18 | NVIDIA L40S | 2.11.0+cu128 / 5.21.0 / 0.21.2 | 1,327,104 |
| G2 | 13 | NVIDIA L40S | 2.11.0+cu128 / 5.21.0 / 0.21.2 | 1,327,104 |
| G3 | 18 | NVIDIA A40 | 2.11.0+cu128 / 5.21.0 / 0.21.2 | 1,241,856–1,276,416 |
| G4 | 13 | NVIDIA A40 | 2.11.0+cu128 / 5.21.0 / 0.21.2 | 1,326,336–1,327,104 |
| G7 | 13 | NVIDIA L40S | 2.4.0+cu121 / 4.52.4 / 0.17.1 | 1,327,104 |
| G9 | 13 | NVIDIA GeForce RTX 3050 Ti Laptop GPU | 2.11.0+cu128 / 5.20.2 / 0.21.2 | 1,241,856–1,267,200 |
| G5 | 13 | NVIDIA GeForce RTX 4090 | 2.8.0+cu128 / 4.45.1 / 0.12.0 | 1,281,024–1,352,448 |
| G6 | 13 | NVIDIA GeForce RTX 4090 | 2.8.0+cu128 / 4.45.1 / 0.12.0 | 1,327,104 |

## G5 / G6: independent protocol (not pairable with the table above)

| Task | G5 | G6 | G5 lr | G6 lr |
|---|---|---|---|---|
| SST-2 (acc) | 93.92 ± 0.23 | 93.43 ± 0.73 | 0.0004 | 0.0004 |
| MRPC (F1) | 91.37 ± 0.95 | 91.68 ± 1.17 | 0.0004 | 0.0004 |
| RTE (acc) | 78.84 ± 2.00 | 78.84 ± 0.91 | 0.0001 | 0.0001 |

## Learning-rate controls (outside the shared protocol)

| Run | lr | Result | This − shared-lr run | G3 − this |
|---|---|---|---|---|
| G1 lowres sst2 | 0.001 | 91.47 ± 0.66 | +0.02 [-0.72, +0.77] | -2.66 [-3.22, -2.11] * |

## H4: subspace overlap of cross-task EVA (G9) with same-task EVA (G3)

Mean squared cosine of the principal angles between the two lora_A row spaces, averaged over the 72 layers; 1 = identical, at most about 0.02 for random subspaces of these ranks (r ≤ 16 in d ≥ 768).

| Target task | Calibrated on | Seeds | Overlap (mean ± sd) | Range | G9 − G3 accuracy |
|---|---|---|---|---|---|
| sst2 | mrpc | 3 | 0.493 ± 0.013 | 0.481–0.507 | -0.19 [-1.63, +1.24] |
| mrpc | rte | 5 | 0.662 ± 0.015 | 0.644–0.683 | -0.52 [-1.21, +0.17] |
| rte | mrpc | 5 | 0.662 ± 0.015 | 0.644–0.683 | +0.87 [-3.31, +5.04] |

The MRPC and RTE rows compare the same two subspaces (EVA on MRPC vs EVA on RTE) and agree to 1e-05 per seed, so there are two independent pairs here, not three.
