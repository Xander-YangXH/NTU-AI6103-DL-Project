"""Report figures from the same run JSONs as collect.py.

    python summary/figures.py      # from the repository root; writes summary/figures/*.png and *.pdf

step0_vs_accuracy: one panel per task; x = mean step-0 ||dL/dB|| (log), y = paired accuracy difference to G1
  with its 95% CI. Groups are named next to their marks; only G5 (GW-EVA) is coloured.
lowres_lr: low-resource accuracy against the learning rate for G1, G3 (EVA) and G8, mean with 95% CI over seeds,
  from the shared runs and the controls run on L40S (G3's shared A40 run is left out so one GPU is compared).
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402

OUT = os.path.join(collect.OUT, "figures")
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # reference palette slots 1-3 (validated all-pairs, light)
TASKS = [("main", "sst2", "SST-2 (acc)"), ("main", "mrpc", "MRPC (F1)"), ("main", "rte", "RTE (acc)")]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                     "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titlecolor": INK, "savefig.dpi": 200})


def scores():
    score, step0 = defaultdict(dict), defaultdict(list)
    for r in collect.load():
        score[(r["group"], r["stage"], r["task"])][r["seed"]] = r["primary"]
        if r["step0_grad_B"] is not None:
            step0[(r["group"], r["stage"], r["task"])].append(r["step0_grad_B"])
    return score, step0


def place_labels(ax, items, min_gap=9.0):
    """Name each mark to its right, nudging labels apart vertically (in points) so none overlap."""
    items = sorted(items, key=lambda it: ax.transData.transform((it[1], it[2]))[1])
    placed = []   # (x_px, y_px) of label anchors already used
    for name, x, y, color in items:
        px, py = ax.transData.transform((x, y))
        ty = py
        for qx, qy in placed:
            if abs(qx - px) < 40 and ty - qy < min_gap * fig_dpi(ax) / 72:
                ty = qy + min_gap * fig_dpi(ax) / 72
        placed.append((px, ty))
        dy = (ty - py) * 72 / fig_dpi(ax)
        ax.annotate(name, (x, y), xytext=(5, dy), textcoords="offset points", va="center", fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.5, shrinkA=0, shrinkB=3) if abs(dy) > 4 else None)


def fig_dpi(ax):
    return ax.figure.dpi


def step0_vs_accuracy(score, step0):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    ticks = {"sst2": [2, 5, 10, 20, 50], "mrpc": [0.2, 0.5, 1, 2, 5], "rte": [0.2, 0.5, 1, 2]}
    for ax, (stage, task, label) in zip(axes, TASKS):
        lo_clip = -6.5   # G7 on RTE (-17) would flatten the panel; it is drawn at the edge and labelled
        labels = []
        for g in collect.SHARED:
            if (g, stage, task) not in step0:
                continue
            x = sum(step0[(g, stage, task)]) / len(step0[(g, stage, task)])
            if g == "G1":
                y, lo, hi = 0.0, 0.0, 0.0
            else:
                _, y, lo, hi = collect.paired(score[("G1", stage, task)], score[(g, stage, task)])
            color = SERIES[0] if g == "G5" else INK_2
            clipped = y < lo_clip
            yy = lo_clip if clipped else y
            if not clipped:
                ax.plot([x, x], [max(lo, lo_clip), hi], color=color, lw=1.2, alpha=0.55, solid_capstyle="round")
            ax.plot(x, yy, marker="v" if clipped else "o", ms=5.5, color=color, mec="white", mew=1.0, zorder=3)
            labels.append((f"{g} ({y:+.1f})" if clipped else g, x, yy, INK if g == "G5" else INK_2))
        ax.axhline(0, color=INK_3, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks[task]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        lo_x, hi_x = ax.get_xlim()
        ax.set_xlim(lo_x, hi_x * 1.6)   # room for the labels right of the last marks
        fig.canvas.draw()
        place_labels(ax, labels)
        ax.set_title(label)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("step-0 ‖∂L/∂B‖ (log)")
    axes[0].set_ylabel("accuracy − G1 (pp, 95% CI)")
    save(fig, "step0_vs_accuracy")


def lowres_lr(score):
    runs = defaultdict(dict)   # label -> lr -> {seed: primary}
    paths = (glob.glob(os.path.join(collect.ROOT, "*", "results", "lowres", "*.json"))
             + glob.glob(os.path.join(collect.ROOT, "*", "results", "control*", "lowres", "*.json")))
    for path in paths:
        r = json.load(open(path))
        if r.get("group") not in ("G1", "G3", "G8") or "GeForce" in str(r.get("gpu")):
            continue
        if r["group"] == "G3" and ("A40" in str(r.get("gpu")) or (r.get("cfg") or {}).get("rho") != 2.0):
            continue   # same GPU (L40S) only; the uniform-rank control is not part of this curve
        runs[r["group"]][r["lr"]] = {**runs[r["group"]].get(r["lr"], {}), r["seed"]: r["final"]["primary"]}
    fig, ax = plt.subplots(figsize=(3.3, 2.8), constrained_layout=True)
    names = {"G1": "G1 LoRA", "G3": "G3 EVA", "G8": "G8 EVA-generic"}
    dodge = {"G1": 0.96, "G3": 1.0, "G8": 1.04}   # small horizontal offsets so the error bars do not overlap
    for color, g in zip(SERIES, ("G1", "G3", "G8")):
        lrs = sorted(runs.get(g, {}))
        if not lrs:
            continue
        ms, his = [], []
        for lr in lrs:
            v = [100 * x for x in runs[g][lr].values()]
            m, sd = collect.mean_sd(v)
            ms.append(m)
            his.append(collect.T95[len(v) - 1] * sd / math.sqrt(len(v)))
        xs = [lr * dodge[g] for lr in lrs]
        ax.errorbar(xs, ms, yerr=his, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.0, capsize=0,
                    elinewidth=1.2, label=names[g])
        ax.annotate(names[g], (xs[-1], ms[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([2e-4, 5e-4, 1e-3], ["2e-4", "5e-4", "1e-3"])
    ax.minorticks_off()
    ax.set_xlim(1.6e-4, 2.2e-3)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("SST-2 1k accuracy (%, 95% CI)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.5)
    save(fig, "lowres_lr")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{png,pdf}"))


if __name__ == "__main__":
    s, g = scores()
    step0_vs_accuracy(s, g)
    lowres_lr(s)
