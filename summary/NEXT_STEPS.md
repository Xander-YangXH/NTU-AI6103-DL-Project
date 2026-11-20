<p align="right"><b>English</b> | <a href="NEXT_STEPS.zh-CN.md">中文</a></p>

# Next steps

Written 2026-11-05, from the results in [`SUMMARY.md`](SUMMARY.md).

**Status on 2026-11-20:**

- **Step 2 is done.** hanyanze reran G5/G6 under the shared settings (11-18; low resource with 3 seeds). GW-EVA does not beat EVA: G5 − G3 is non-significant on every task, with slightly negative means (see [`SUMMARY.md`](SUMMARY.md) §2). On the decision table below, the story is now the analysis / negative-result one.
- **Step 1 is partly done.** G9's cross-task overlap exists as a layer average (zjz, 11-07). The H1 measurement itself (GW-EVA vs EVA subspaces) and the noise floor are still missing.
- **Step 5 is partly answered from existing data.** `collect.py` now compares the first-500-step training loss at equal learning rates (SUMMARY finding 3): data-aware inits fit faster early on MRPC, GW-EVA is no faster than EVA, and in low resource EVA/GW-EVA fit more slowly in steps 51–200. Evaluating on the dev/val split at fixed step budgets still needs the `train.py` option.
- **Step 4 is done** (11-20, `G1G2/results/control*/`): EVA prefers a lower learning rate in low resource (lr 2e-4: 90.87 vs 89.50 at 1e-3) and then trails LoRA by only 0.57; uniform ranks (ρ = 1) change nothing. Still open: the same learning-rate check for G5/G6 and G8.
- **Step 7 is done.** eddy ran G8 (13 main + 5 low-resource runs, branch `eddy_G8`): generic-text calibration matches task-data EVA in accuracy, with subspace overlap 0.64–0.72.
- **Noise floor done (part of step 1)**: EVA against itself on a disjoint sample of the same task overlaps 0.92–0.95, so G8/G9's 0.49–0.72 are real subspace changes. The H1 measurement itself is still open (hanyanze).
- **Step 3 is started.** G1 and G3 now have ten seeds on MRPC, RTE and low resource; G5 still needs seeds 5–9.
- One G1 RTE seed (of ten) collapsed to a single class; G3's ten RTE seeds did not. This bears on step 6.
- Every other item below is still open.

## Where we stand

- On the full-data tasks, no initialisation (G2, G3, G4, G7, G9) beats plain LoRA (G1) significantly.
- In the low-resource setting, EVA (G3) is significantly **worse** than LoRA (−2.66 points at equal lr).
- **GW-EVA (our method) has not been tested yet.** The only G5/G6 runs follow an independent protocol and cannot be paired with G1/G3.
- G9 shows that calibrating EVA on another task does not change accuracy, even though the subspace overlap with same-task EVA is only 0.49–0.66. Accuracy is insensitive to the directions, so it cannot decide H4; the overlap needs a same-task noise floor before it can. Whether Σ_g's subspace differs from EVA's is H1, and nobody has measured it yet.

The priority is therefore to find out **whether GW-EVA works**, not to fill in the remaining groups.

Compute is cheap: under the shared protocol one run takes about 1–4 min on MRPC / RTE and 3–8 min on SST-2, so a full G5 pass is roughly 2 GPU hours.

## P0: this week (decides the paper's story)

### 1. H1 diagnostic (owner: hanyanze)

The code already exists. `G5G6/src/gw_eva/calibrate.py` computes the subspace overlap and principal angles between `eva_centered`, `activation_uncentered` and `gw_uncentered`, and the Pearson correlation of ‖g‖² with ‖x‖². The formal runs only enabled `gw_uncentered` (`study.py:117` sets `modes=['gw_uncentered']`), so no H1 numbers exist yet.

- Turn on all three modes and calibrate with the shared settings: max length 128, 1,024 examples, on SST-2, MRPC, RTE and the low-resource 1k subset, seeds 0–2.
- Report per layer: overlap of the top-8 subspaces (EVA vs GW-EVA), principal angles, and corr(‖g‖², ‖x‖²). Summarise by layer type (q/k/v/o/FFN) and depth.
- **Noise floor (required):** also compute the overlap of EVA with itself on two disjoint calibration samples. EVA and GW-EVA count as different only if their overlap is clearly below this floor.
- **Cross-task overlap (for H4):** also compute the overlap of EVA calibrated on task X with EVA calibrated on the G9 source task. This turns G9's accuracy result into a direct measurement. *Done by zjz on 2026-11-07 as a layer average (0.49 MRPC vs SST-2, 0.66 RTE vs MRPC); still needed: the noise floor above, the per-layer values, and the SST-2 vs RTE pair.*
- Output: `G5G6/results/h1/*.json` plus one figure of overlap by layer.

### 2. G5 / G6 under the shared protocol (owners: hanyanze + yinchenyu) — done 2026-11-18

- Plug the GW-EVA initialisation into the shared pipeline in `G3G4/src/train.py`, keeping everything else identical to G3 (split, head warm-up, epochs, lr grid, seeds, budget).
- Run tune (3 lr × 3 tasks, seed 0), main (SST-2 3 seeds, MRPC / RTE 5 seeds) and low-resource (5 seeds) for G5; then the same for G6.
- Write JSONs in the shared format to `G5G6/results/{tune,main,lowres}/` with `shared_protocol_compliant = True`; move the current independent-protocol runs to a separate folder (e.g. `G5G6/results/independent/`) so `collect.py` does not mix them.
- Estimate: G5 about 2 GPU hours, G6 about 1.5 GPU hours.

## P1: make the conclusions trustworthy (all cheap)

### 3. More seeds (owners: yxh for G1, yinchenyu for G3, hanyanze for G5)

The RTE intervals are ±3–4 points wide and can only detect large effects. Extend G1, G3 and G5 to **10 seeds** on MRPC, RTE and low-resource (seeds 5–9, same lr). About 45 runs, around 2 GPU hours in total.

### 4. Why EVA hurts in low resource (owner: yinchenyu) — done 2026-11-20 (run by yxh)

Per-epoch validation accuracy, mean of 5 seeds:

| Epoch | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| G1 @ 1e-3 | 79.4 | **90.2** | 90.3 | 89.7 | 90.4 | 91.5 |
| G3 @ 1e-3 | 79.5 | 80.0 | 86.9 | 86.6 | 88.3 | 88.8 |

Both start equally, then G1 jumps to 90 while G3 stalls around 80. EVA is not converging faster; something in optimisation goes wrong. One hypothesis: a 4–9× larger initial gradient combined with lr 1e-3 (inherited from full SST-2) makes the effective step too large.

Controls, 5 seeds each, stored under `G3G4/results/control/lowres/` (which `collect.py` already reads):

- G3 low-resource at lr 5e-4 and 2e-4;
- EVA directions with uniform ranks: G3 with `rho = 1.0`, so every layer keeps r = 8 and the budget is exact. This separates direction from rank redistribution.

About 15 runs, about 20 GPU minutes.

### 5. Early-training metrics (owner: yinchenyu for the shared `train.py`; yxh for the analysis)

Initialisation helps only in the first few dozen steps. Mean loss, all from existing JSONs:

| Setting | G1 first 50 | G3 first 50 | G1 first 500 | G3 first 500 |
|---|---|---|---|---|
| SST-2 | 0.494 | **0.443** | **0.327** | 0.350 |
| Low-res SST-2 | 0.603 | **0.535** | **0.140** | 0.172 |
| MRPC | 0.607 | **0.591** | 0.396 | **0.355** |

(Learning rates differ between groups here, e.g. SST-2 G1 5e-4 vs G3 1e-3, so treat this as indicative.)

- Add an option to `train.py` that evaluates on the dev/val split at fixed step budgets (100, 250, 500 steps) and stores them in the JSON (e.g. `per_step_eval`).
- Compare G1, G3 and G5 at **equal lr** at those budgets. This matches H2's actual wording ("lower loss over the first 500 steps").
- Add the early-loss table to `summary/collect.py`.

### 6. G7 collapse on RTE (owner: liyouran)

RTE seed 1 predicts one class for all 277 examples. Check whether this comes from LoRA-GA rewriting W (e.g. the size of the W offset, the loss curve and gradient norm of the collapsed seed), and try a few extra seeds or a lower lr. The goal is an explanation for the report, not necessarily a fix.

## P2: side threads, if time allows

### 7. G8 EVA-generic — done 2026-11-20 (eddy)

Same code as G9 with the calibration set swapped to 1,024 wikitext-103 segments (length 128). Reuse G3's learning rates, 13 main runs. Together with G9 this completes the H4 line.

### 8. Start writing the report now (owner: everyone; outline by yxh)

Method, theory (EVA, GW-EVA and LoRA-GA as solutions of one objective under different assumptions) and experimental setup are fixed and can be written now. Each group owner writes the paragraph and table for their group.

## Decision point after P0

| H1 | G5 vs G3 (shared protocol) | Story |
|---|---|---|
| Subspaces differ | G5 better | **Method paper**: GW-EVA improves on EVA, explained by H1 |
| Subspaces differ | No difference | **Analysis paper**: the gradient-weighted subspace is different but does not matter for final accuracy; focus on early-training effects |
| Subspaces almost identical | – | **Negative result**: EVA's independence assumption holds in practice, so GW-EVA reduces to EVA |

The analysis / negative-result versions are complete papers too. Core evidence: the magnitude vs direction split (G2), early-only benefits, the low-resource optimisation problem, G9's task-insensitivity and the unified theory.

## Summary of owners

| Task | Owner | Priority | Compute |
|---|---|---|---|
| H1 diagnostic | hanyanze | P0 | < 1 GPU hour |
| G5 / G6 under shared protocol | hanyanze + yinchenyu | P0 | ~3.5 GPU hours |
| 10 seeds for G1 / G3 / G5 | yxh / yinchenyu / hanyanze | P1 | ~2 GPU hours |
| Low-resource controls | yinchenyu | P1 | ~20 GPU minutes |
| Early-training metrics | yinchenyu + yxh | P1 | reruns as needed |
| G7 RTE collapse | liyouran | P1 | small |
| G8 | to be assigned (zjz suggested) | P2 | ~1 GPU hour |
| Report | everyone | ongoing | – |

After each step, push results to your own branch; they will be merged into `summary` and `collect.py` rerun.
