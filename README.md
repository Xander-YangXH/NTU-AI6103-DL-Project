<p align="right"><b>English</b> | <a href="README.zh-CN.md">中文</a></p>

# GW-EVA: Gradient-Weighted LoRA Initialisation Without the Independence Assumption

AI6103 Deep Learning group project (Option 2: improve on an existing codebase), team of six.

- **Base paper**: EVA (Paischer et al., NeurIPS 2025). It initialises LoRA's A matrix with the principal components of layer activations and redistributes ranks across layers by explained variance.
- **Codebase**: the authors' official repository (MIT licence) and the EVA implementation in Hugging Face PEFT.
- **Our change**: we drop one independence assumption from EVA's optimality proof, derive a gradient-weighted initialisation, **GW-EVA**, and implement it on top of PEFT.

## 1. Research question

EVA's optimality result (Theorem 3.2) assumes that the layer-output gradient $g = \partial L / \partial y$ is statistically independent of the input activation $x$.

Without this assumption, the gradient signal that B receives at initialisation (B = 0) is exactly:

$$
\mathbb{E}\left\Vert \frac{\partial L}{\partial B}\right\Vert _F^2 = \mathrm{Tr}\left(A\,\Sigma_g\,A^\top\right),\qquad
\Sigma_g = \mathbb{E}\left[\Vert g\Vert^2\, x x^\top\right]
$$

The optimal A is therefore the top-r eigenvectors of $\Sigma_g$, which we call **GW-EVA**. When $\Vert g\Vert^2$ is independent of $x$, it reduces to EVA.

For a mini-batch of size m:

$$
\mathbb{E}\left\Vert \bar G_m A^\top\right\Vert _F^2 = \tfrac{1}{m}\mathrm{Tr}\left(A\Sigma_g A^\top\right) + \left(1-\tfrac{1}{m}\right)\left\Vert \mathbb{E}[g x^\top]A^\top\right\Vert _F^2
$$

The direction that maximises the second term is exactly the mean-gradient direction used by LoRA-GA. EVA, GW-EVA and LoRA-GA are therefore solutions of the same objective under different assumptions.

A second thread asks how much task information a LoRA initialisation needs. Ordered from least to most task information, four initialisations form a line:

generic corpus (G8) → another task's inputs (G9) → this task's inputs (G3, EVA) → this task's inputs + label gradients (G5, GW-EVA)

## 2. Hypotheses

| | Hypothesis | If it fails |
|---|---|---|
| **H1** | In RoBERTa-base, $\Vert g\Vert^2$ is correlated with $x$, and the top-r subspaces of $\Sigma_g$ and of the activation covariance differ measurably (week 1, no training needed) | Empirical support for EVA's assumption; GW-EVA reduces to EVA and we report a negative result |
| **H2** | At equal parameter count, GW-EVA has a larger step-0 gradient, lower loss over the first 500 steps and higher accuracy than EVA, with a paired 95% CI excluding 0 at least in the low-resource setting | The extra backward pass is not worth it |
| **H3** | Allocating ranks by explained variance per parameter, instead of per rank, changes the result under a fixed parameter budget (a controlled factor, not a novelty claim) | EVA's original per-rank allocation is sufficient |
| **H4** | EVA's activation subspace carries task-specific information: calibrating on a generic corpus or another task changes both the subspace and the accuracy | Activation statistics are mostly determined by the model, so task information can only enter through label gradients, which strengthens the case for GW-EVA |

## 3. Experimental groups

The core is a 2×2 factorial design, direction × allocation (G3–G6), plus reference groups.

| Group | Direction of A | Rank allocation | Question answered | Folder | Status |
|---|---|---|---|---|---|
| G1 LoRA | Kaiming random | Uniform r = 8 | Baseline | [`G1G2/`](G1G2/) | Done |
| G2 magnitude control | Random orthonormal, row norms matched to EVA | Uniform r = 8 | Do gains come from direction or magnitude? | [`G1G2/`](G1G2/) | Done |
| G3 EVA | Activation covariance | By rank count (original, ρ = 2) | Reproduce the original method | [`G3G4/`](G3G4/) | Done |
| G4 | Activation covariance | By parameter cost | Allocation only (H3) | [`G3G4/`](G3G4/) | Done |
| G5 GW-EVA | $\Sigma_g$ | By rank count | Direction only (H2) | [`G5G6/`](G5G6/) | Done (shared settings; low-resource 3 seeds) |
| G6 GW-EVA | $\Sigma_g$ | By parameter cost | Direction and allocation combined | [`G5G6/`](G5G6/) | Done (shared settings; low-resource 3 seeds) |
| G7 LoRA-GA | Mean gradient (B ≠ 0, rewrites W) | Uniform r = 8 | Representative of the second term | [`G7/`](G7/) | Done |
| G8 EVA-generic | Activation covariance on a generic corpus (wikitext-103) | By rank count | Task-agnostic end (H4) | [`G8/`](G8/) | Done (incl. low-resource) |
| G9 EVA-cross | Activation covariance from another task (SST-2←MRPC, MRPC←RTE, RTE←MRPC) | By rank count | How task-specific are the inputs? (H4) | [`G9/`](G9/) | Done |

**Model and tasks**: RoBERTa-base on GLUE SST-2, MRPC and RTE, plus a 1,000-example low-resource SST-2 subset (G1, G3, G5, G6).

**LoRA placement and budget**: all 72 attention and FFN linear layers, with a fixed budget of **1,327,104** trainable parameters (the count for uniform r = 8).

## 4. Shared experimental settings

Every group must use these settings so that runs can be compared pairwise by seed. The reference implementation is `data.py` and `train.py` in `G3G4/src/`; G1/G2 reuse them directly.

| Setting | Value |
|---|---|
| Learning-rate grid | 2×10⁻⁴, 5×10⁻⁴, 1×10⁻³ |
| Seeds | 0–2 for SST-2, 0–4 for MRPC / RTE; seed 0 for tuning |
| Epochs | SST-2 3, MRPC 10, RTE 10, low-resource 20 |
| Batch size / max length / warm-up ratio / weight decay | 32 / 128 / 6% / 0; bf16 mixed precision; no gradient clipping |
| LoRA α / dropout | α = 8 (α/r = 1) / 0; after EVA's redistribution PEFT scales α with the rank so that α/r is unchanged |
| Classification-head warm-up | Encoder frozen, 1 epoch, learning rate 1×10⁻³ |
| Dev split | 10% of the training set, `SPLIT_SEED = 2026`; the low-resource 1k subset uses seed 2027 |
| Final training data | The remaining 90%, the same as during tuning |
| EVA calibration | 1,024 random training examples, batch 16, labels removed; τ = 0.99, ρ = 2 |
| Selection and reporting | Last epoch; hyperparameters chosen on dev; the official validation set is used only for the final evaluation |
| Low-resource learning rate | Each group reuses the learning rate it chose on full SST-2 |
| Result format | One JSON per run; fields as at the end of `G3G4/src/train.py` |
| Statistics | Same-seed paired differences with 95% confidence intervals (t distribution, df = n − 1) |
| Metrics | Accuracy for SST-2 / RTE, F1 and accuracy for MRPC; step-0 ‖∂L/∂B‖ and loss over the first 500 steps; initialisation time, total time, peak memory |

Learning rates chosen on the dev split (seed 0); low resource reuses each group's full-SST-2 choice:

| Group | SST-2 | MRPC | RTE |
|---|---|---|---|
| G1 | 5e-4 | 2e-4 | 5e-4 |
| G2 | 5e-4 | 1e-3 | 2e-4 |
| G3 | 1e-3 | 2e-4 | 2e-4 |
| G4 | 5e-4 | 2e-4 | 2e-4 |
| G5 | 1e-3 | 2e-4 | 2e-4 |
| G6 | 5e-4 | 2e-4 | 2e-4 |
| G7 | 2e-4 | 2e-4 | 2e-4 |

G8 / G9 reuse G3's learning rates.

## 5. Results

All groups ran under the shared settings above and are paired by seed. Full tables, paired differences, caveats and the open items are in [`summary/SUMMARY.md`](summary/SUMMARY.md), regenerated by `python summary/collect.py`.

Official validation, mean ± sd (%); SST-2 3 seeds, MRPC / RTE 5, low resource 5 (G5/G6: 3):

| Task | G1 LoRA | G2 | G3 EVA | G4 | G5 GW-EVA | G6 | G7 LoRA-GA | G8 | G9 |
|---|---|---|---|---|---|---|---|---|---|
| SST-2 (acc) | 93.92 ± 0.00 | 94.07 ± 0.24 | 94.19 ± 0.43 | 93.65 ± 0.24 | 94.04 ± 0.30 | **94.30 ± 0.24** | 93.96 ± 0.46 | 93.88 ± 0.46 | 94.00 ± 0.63 |
| MRPC (F1) | 91.34 ± 0.43 | 91.16 ± 1.11 | **92.02 ± 0.79** | 91.50 ± 0.68 | 91.14 ± 0.69 | 91.41 ± 0.52 | 91.83 ± 0.46 | 91.11 ± 0.34 | 91.50 ± 0.60 |
| RTE (acc) | 73.94 ± 2.10 | 73.86 ± 1.04 | 74.80 ± 2.49 | 73.57 ± 2.06 | 73.57 ± 2.43 | 72.92 ± 2.09 | 56.68 ± 8.85 | 72.06 ± 2.18 | **75.67 ± 1.70** |
| Low-res SST-2 1k (acc) | **91.44 ± 0.28** | – | 88.81 ± 0.83 | – | 88.46 ± 0.29 | 89.98 ± 0.18 | – | 89.89 ± 0.52 | – |

Report figures: [`summary/figures/`](summary/figures/) (step-0 gradient vs accuracy; low-resource accuracy vs learning rate; H4 subspace overlap by layer), generated by `python summary/figures.py`.

Paired differences (percentage points, 95% CI; `*` = interval excludes 0):

| Comparison | SST-2 | MRPC | RTE | Low resource |
|---|---|---|---|---|
| G5 − G3 (H2) | −0.15 [−1.07, +0.76] | −0.88 [−2.05, +0.29] | −1.23 [−4.09, +1.64] | −0.46 [−3.82, +2.90] |
| G3 − G1 | +0.27 [−0.81, +1.35] | +0.68 [−0.60, +1.97] | +0.87 [−3.27, +5.01] | −2.64 [−3.70, −1.57] * |
| G5 − G1 | +0.11 [−0.64, +0.87] | −0.20 [−1.14, +0.75] | −0.36 [−5.35, +4.63] | −3.10 [−3.59, −2.60] * |
| G7 − G1 | +0.04 [−1.11, +1.19] | +0.49 [−0.44, +1.41] | −17.26 [−27.93, −6.59] * | – |
| G3 − G1, both at lr 2e-4 on one GPU⁵ | – | – | – | −0.57 [−1.45, +0.30] |

| Hypothesis | Verdict |
|---|---|
| H1: Σ_g's subspace differs from EVA's | Not measured yet |
| H2: GW-EVA beats EVA | **Not supported**: G5 − G3 non-significant everywhere, means slightly negative, and no lower early loss, although GW-EVA's step-0 gradient is 1.3–2.1× EVA's |
| H3: cost-based allocation matters | **Not supported**: G4 − G3 and G6 − G5 non-significant; EVA with uniform ranks matches the original in low resource |
| H4: how task-specific the inputs are | **Answered**: EVA calibrated on another task (G9) or on generic wikitext-103 text (G8) matches EVA on task data in accuracy, although its subspace overlaps by only 0.49–0.72 against a same-task noise floor of 0.92–0.95 |

⁵ From the low-resource controls in [`summary/SUMMARY.md`](summary/SUMMARY.md), outside the shared protocol; 2e-4 is EVA's best low-resource learning rate, and G1 gives the same result at every learning rate tried.

Main findings:

- **No initialisation beats plain LoRA on full data.** All groups except G7 are within ±1.9 points of G1, and every interval contains 0. That includes EVA calibrated on another task (G9) or on generic text (G8).
- **Data-aware inits fit faster early, but the gain does not last.** At equal lr on MRPC, EVA, GW-EVA and LoRA-GA all have a significantly lower training loss than LoRA over the first 500 steps, with no difference in final F1. GW-EVA is no faster than EVA at any stage.
- **Larger initial gradients do not buy accuracy.** The step-0 gradient on B grows 7–37× from G1 to the data-aware inits; matching only EVA's magnitude (G2) or its directions (G3) or adding label gradients (G5) leaves accuracy flat.
- **With 1k examples, spectral inits are worse than LoRA only because of the learning rate.** Under the shared rule G3 trails G1 by 2.64 (G5 −3.10, G6 −1.57, G8 −1.56). LoRA is insensitive to the learning rate while EVA prefers a lower one: with both at 2e-4 the gap shrinks to −0.57, not significant. Removing EVA's rank redistribution changes nothing; the loss curves show EVA fitting more slowly in steps 51–200 at the high learning rate.
- **On one GPU and at equal learning rate, EVA helps only on RTE.** The apparent MRPC edge with ten seeds came from mixing GPUs (with 15 seeds all on L40S, G3 − G1 = +0.14, not significant). On RTE at lr 2e-4, G3 − G1 = +2.13 [+0.97, +3.29]; at LoRA's own tuned lr (5e-4) the difference is not significant.
- **RTE is fragile.** LoRA-GA averages 56.68 there (one seed predicts a single class, two are near chance), but plain LoRA also collapsed on 1 of 10 seeds once G1 was extended to ten; see [`summary/SUMMARY.md`](summary/SUMMARY.md) §2, finding 7.

## 6. Repository layout

Each member worked on their own branch with one folder per pair of groups, holding code, Slurm scripts and result JSONs; `main` now merges all of them. Large files (model weights, data, EVA caches, logs) stay out of git via `.gitignore`.

| Folder | Contents | Source branch |
|---|---|---|
| `G1G2/` | G1 LoRA baseline, G2 magnitude control, G1 extra seeds, low-resource controls for G1 and G3 | `yxh` |
| `G3G4/` | G3 EVA reproduction, G4 parameter-cost allocation; the shared data split and training pipeline | `yinchenyu_g3g4` |
| `G5G6/` | G5 / G6 GW-EVA under the shared settings | `hanyanze` |
| `G7/` | G7 LoRA-GA | `liyouran` |
| `G8/` | G8 EVA-generic (wikitext-103 calibration in G3's token layout) and its subspace overlap with G3 | `eddy_G8` |
| `G9/` | G9 EVA-cross and the H4 subspace overlap | `zjz_g9` |
| `summary/` | Cross-group tables (`collect.py`), findings and next steps | `summary` |

## 7. Compute

CCDS GPU Cluster TC2 with Slurm. Each member's QoS allows 1 GPU, at most 2 jobs, 30 GB of memory and 6 hours per job. Each job packs several training runs and runs 2 processes side by side on one GPU. Every run writes its JSON as soon as it finishes, so resubmitting a job resumes it.

## 8. Course requirements in brief

- Option 2: pick a paper from 2019 or later with public code and implement an improvement on top of it; reusing the original code is allowed.
- Do not repeat a model–dataset combination used in any member's individual project.
- Do not artificially lower baseline performance.
- Scoring, 30 points in total: novelty and difficulty 10, technical approach 10, writing 10.
