"""Per-layer rank heatmaps of G3 (EVA) and G4 (parameter-cost allocation) for the report (H3).

    python jobs/summarize.py        # writes results/g34_ranks.csv
    python jobs/plot_ranks.py       # writes results/g34_rank_heatmap.{png,pdf} and g34_rank_heatmap_diff.{png,pdf}
Rows are encoder layers, columns are the six LoRA targets; each cell is the rank averaged over the main-stage
seeds. A layer whose LoRA was removed (rank 0) is missing from the CSV and is drawn as 0.
The second figure shows G4 - G3 per cell (red: G4 gives the layer more rank, blue: fewer).
"""
import csv
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KINDS = ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]
LABELS = ["Q", "K", "V", "Attn-out", "FFN-in", "FFN-out"]
TASKS = ["sst2", "mrpc", "rte"]
GROUPS = ["G3", "G4"]
CAP = 16   # r * rho, the per-layer maximum for both groups

rows = [r for r in csv.DictReader(open("results/g34_ranks.csv")) if r["run_id"].startswith("main_")]
if not rows:
    raise SystemExit("no main-stage rows in results/g34_ranks.csv: run jobs/summarize.py after the main stage")
n_layers = 1 + max(int(re.search(r"layer\.(\d+)\.", r["layer"]).group(1)) for r in rows)

grids = {}
for g in GROUPS:
    for t in TASKS:
        seeds = defaultdict(lambda: np.zeros((n_layers, len(KINDS))))
        for r in rows:
            if r["group"] == g and r["task"] == t:
                i = int(re.search(r"layer\.(\d+)\.", r["layer"]).group(1))
                seeds[r["seed"]][i, KINDS.index(r["kind"])] = int(r["rank"])
        if seeds:
            grids[(g, t)] = np.mean(list(seeds.values()), axis=0)

fig, axes = plt.subplots(len(GROUPS), len(TASKS), figsize=(3.6 * len(TASKS), 0.32 * n_layers * len(GROUPS) + 1.6),
                         squeeze=False, constrained_layout=True)
for a, g in enumerate(GROUPS):
    for b, t in enumerate(TASKS):
        ax = axes[a][b]
        if (g, t) not in grids:
            ax.set_axis_off()
            continue
        m = grids[(g, t)]
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=CAP, aspect="auto")
        for i in range(n_layers):
            for j in range(len(KINDS)):
                ax.text(j, i, f"{m[i, j]:.0f}" if abs(m[i, j] - round(m[i, j])) < 0.05 else f"{m[i, j]:.1f}",
                        ha="center", va="center", fontsize=7, color="white" if m[i, j] > CAP * 0.6 else "black")
        ax.set_xticks(range(len(KINDS)), LABELS, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n_layers), [str(i) for i in range(n_layers)], fontsize=8)
        attn, ffn = m[:, :4].sum(), m[:, 4:].sum()
        ax.set_title(f"{g} {t.upper()}  (attn {attn:.0f} / FFN {ffn:.0f} ranks)", fontsize=9)
        if b == 0:
            ax.set_ylabel("encoder layer")
fig.colorbar(im, ax=axes, shrink=0.6, label="rank (mean over main seeds)")
for ext in ("png", "pdf"):
    fig.savefig(f"results/g34_rank_heatmap.{ext}", dpi=200)
print("wrote results/g34_rank_heatmap.png and .pdf")

# G4 - G3, same task and layer, both averaged over the main-stage seeds
tasks = [t for t in TASKS if ("G3", t) in grids and ("G4", t) in grids]
diffs = {t: grids[("G4", t)] - grids[("G3", t)] for t in tasks}
lim = max(1.0, max(float(np.abs(d).max()) for d in diffs.values()))
fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 0.32 * n_layers + 1.6),
                         squeeze=False, constrained_layout=True)
for b, t in enumerate(tasks):
    ax, d = axes[0][b], diffs[t]
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(n_layers):
        for j in range(len(KINDS)):
            v = d[i, j]
            label = "0" if abs(v) < 0.05 else f"{v:+.0f}" if abs(v - round(v)) < 0.05 else f"{v:+.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > lim * 0.6 else "black")
    ax.set_xticks(range(len(KINDS)), LABELS, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_layers), [str(i) for i in range(n_layers)], fontsize=8)
    ax.set_title(f"G4 \u2212 G3 {t.upper()}  (attn {d[:, :4].sum():+.0f} / FFN {d[:, 4:].sum():+.0f} ranks)", fontsize=9)
    if b == 0:
        ax.set_ylabel("encoder layer")
fig.colorbar(im, ax=axes, shrink=0.8, label="rank difference G4 \u2212 G3")
for ext in ("png", "pdf"):
    fig.savefig(f"results/g34_rank_heatmap_diff.{ext}", dpi=200)
print("wrote results/g34_rank_heatmap_diff.png and .pdf")
