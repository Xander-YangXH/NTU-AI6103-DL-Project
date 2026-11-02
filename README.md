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

| Group | Direction of A | Rank allocation | Question answered | Branch | Status |
|---|---|---|---|---|---|
| G1 LoRA | Kaiming random | Uniform r = 8 | Baseline | [`yxh`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yxh/G1G2) | Done |
| G2 magnitude control | Random orthonormal, row norms matched to EVA | Uniform r = 8 | Do gains come from direction or magnitude? | [`yxh`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yxh/G1G2) | Done |
| G3 EVA | Activation covariance | By rank count (original, ρ = 2) | Reproduce the original method | [`yinchenyu_g3g4`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4) | Done |
| G4 | Activation covariance | By parameter cost | Allocation only (H3) | [`yinchenyu_g3g4`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4) | Done |
| G5 GW-EVA | $\Sigma_g$ | By rank count | Direction only (H2) | [`hanyanze`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/hanyanze/G5G6) | 13 runs complete (independent protocol; not a shared-protocol rerun) |
| G6 GW-EVA | $\Sigma_g$ | By parameter cost | Direction and allocation combined | [`hanyanze`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/hanyanze/G5G6) | 13 runs complete (independent protocol; not a shared-protocol rerun) |
| G7 LoRA-GA | Mean gradient (B ≠ 0, rewrites W) | Uniform r = 8 | Representative of the second term | – | Not started |
| G8 EVA-generic | Activation covariance on a generic corpus (wikitext-103) | By rank count | Task-agnostic end (H4) | – | Not started |
| G9 EVA-cross | Activation covariance from another task (SST-2←MRPC, MRPC←RTE, RTE←MRPC) | By rank count | How task-specific are the inputs? (H4) | – | Not started |

**Model and tasks**: RoBERTa-base on GLUE SST-2, MRPC and RTE, plus a 1,000-example low-resource SST-2 subset (G1, G3, G5).

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

Learning rates chosen so far:

| Group | SST-2 | MRPC | RTE |
|---|---|---|---|
| G1 | 5e-4 | 2e-4 | 5e-4 |
| G2 | 5e-4 | 1e-3 | 2e-4 |
| G3 | 1e-3 | 2e-4 | 2e-4 |
| G4 | 5e-4 | 2e-4 | 2e-4 |

G8 / G9 reuse G3's learning rates.

## 5. Results so far

**G3 / G4** (official validation, mean ± sd, %):

| Task | Seeds | G3 | G4 | G4 − G3 (95% CI) |
|---|---|---|---|---|
| SST-2 (acc) | 3 | 94.19 ± 0.43 | 93.65 ± 0.24 | −0.54 [−1.69, +0.62] |
| MRPC (F1) | 5 | 92.02 ± 0.79 | 91.50 ± 0.68 | −0.52 [−1.56, +0.52] |
| RTE (acc) | 5 | 74.80 ± 2.49 | 73.57 ± 2.06 | −1.23 [−4.31, +1.85] |

Allocating by parameter cost (G4) did not help; none of the three differences is significant. G3 on low-resource SST-2 1k: 88.81 ± 0.83. See [`G3G4/README.md`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/tree/yinchenyu_g3g4/G3G4) for details.

**G1 / G2** (official validation, mean ± sd, %; paired differences in percentage points with 95% CI):

| Task | Seeds | G1 | G2 | G2 − G1 | G3 − G1 |
|---|---|---|---|---|---|
| SST-2 (acc) | 3 | 93.92 ± 0.00 | 94.07 ± 0.24 | +0.15 [−0.44, +0.75] | +0.27 [−0.81, +1.35] |
| MRPC (F1) | 5 | 91.34 ± 0.43 | 91.16 ± 1.11 | −0.17 [−1.59, +1.24] | +0.68 [−0.60, +1.97] |
| RTE (acc) | 5 | 73.94 ± 2.10 | 73.86 ± 1.04 | −0.07 [−3.23, +3.09] | +0.87 [−3.27, +5.01] |
| Low-resource SST-2 1k | 5 | 91.44 ± 0.28 | – | – | −2.64 [−3.70, −1.57] |

- Matching EVA's row norm (G2) raises the step-0 gradient on B by 1.8× but does not change accuracy: magnitude alone brings no gain.
- At equal row norm, EVA's directions (G3) give a 3.9–8.5× larger step-0 gradient than random ones, yet only a small, non-significant accuracy gain.
- In the low-resource setting G1 beats G3 significantly, but the two use different learning rates (5e-4 vs 1e-3, each reused from full SST-2), so a same-learning-rate control is needed before drawing conclusions.

Details: [`G1G2/PROGRESS.md`](https://github.com/Xander-YangXH/NTU-AI6103-DL-Project/blob/yxh/G1G2/PROGRESS.md).

## 6. Repository layout

- Each member works on their own branch, with one folder per pair of groups (e.g. `G1G2/`, `G3G4/`) holding code, Slurm scripts and result JSONs.
- `main` holds only this overview and the shared conventions.
- Large files (model weights, data, EVA caches, logs) stay out of git via `.gitignore`.

| Branch | Contents |
|---|---|
| `main` | This overview |
| `yinchenyu_g3g4` | G3 EVA reproduction and G4 parameter-cost allocation; includes the shared data split and training pipeline |
| `yxh` | G1 LoRA baseline and G2 magnitude control |
| `hanyanze` | G5/G6 gradient-weighted directions and cost allocation; independent protocol, 26 main runs |

## 7. Compute

CCDS GPU Cluster TC2 with Slurm. Each member's QoS allows 1 GPU, at most 2 jobs, 30 GB of memory and 6 hours per job. Each job packs several training runs and runs 2 processes side by side on one GPU. Every run writes its JSON as soon as it finishes, so resubmitting a job resumes it.

## 8. Course requirements in brief

- Option 2: pick a paper from 2019 or later with public code and implement an improvement on top of it; reusing the original code is allowed.
- Do not repeat a model–dataset combination used in any member's individual project.
- Do not artificially lower baseline performance.
- Scoring, 30 points in total: novelty and difficulty 10, technical approach 10, writing 10.
