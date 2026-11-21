"""Report version of Figure 2 (step0_vs_accuracy) with hand-placed labels where marks crowd.

    python report/figures/make_step0_vs_accuracy.py . report/figures   # from the repository root of a branch that has summary/ (e.g. main)
Same data and drawing as summary/figures.py:step0_vs_accuracy; only the y label and the label positions differ.
"""
import os
import sys

repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "summary"))
os.chdir(repo)
import figures as F  # noqa: E402  (summary/figures.py: palette, rcParams, collect)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

collect = F.collect
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
# (task, group) -> label offset in points from the mark; groups not listed use the default (6, 0)
OFFSETS = {
    ("sst2", "G6"): (6, 6), ("sst2", "G3"): (-16, 12), ("sst2", "G9"): (-22, 2),
    ("sst2", "G8"): (-20, -12), ("sst2", "G4"): (6, -6), ("sst2", "G5"): (7, -3),
    ("mrpc", "G4"): (-7, 0), ("mrpc", "G9"): (-8, 12), ("mrpc", "G6"): (7, 7), ("mrpc", "G5"): (8, -3),
    ("mrpc", "G8"): (-18, -9),
    ("rte", "G9"): (6, 7), ("rte", "G3"): (6, -1), ("rte", "G4"): (-18, -6), ("rte", "G5"): (7, 6),
    ("rte", "G6"): (7, -2), ("rte", "G8"): (6, -5),
}


def step0_vs_accuracy(score, step0):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    ticks = {"sst2": [2, 5, 10, 20, 50], "mrpc": [0.2, 0.5, 1, 2, 5], "rte": [0.2, 0.5, 1, 2]}
    for ax, (stage, task, label) in zip(axes, F.TASKS):
        lo_clip = -6.5
        for g in collect.SHARED:
            if (g, stage, task) not in step0:
                continue
            x = sum(step0[(g, stage, task)]) / len(step0[(g, stage, task)])
            if g == "G1":
                y, lo, hi = 0.0, 0.0, 0.0
            else:
                _, y, lo, hi = collect.paired(score[("G1", stage, task)], score[(g, stage, task)])
            color = F.SERIES[0] if g == "G5" else F.INK_2
            clipped = y < lo_clip
            yy = lo_clip if clipped else y
            if not clipped:
                ax.plot([x, x], [max(lo, lo_clip), hi], color=color, lw=1.2, alpha=0.55, solid_capstyle="round")
            ax.plot(x, yy, marker="v" if clipped else "o", ms=5.5, color=color, mec="white", mew=1.0, zorder=3)
            name = f"{g} ({y:+.1f})" if clipped else g
            dx, dy = OFFSETS.get((task, g), (6, 0))
            far = abs(dy) > 4 or dx < 0
            ax.annotate(name, (x, yy), xytext=(dx, dy), textcoords="offset points", va="center",
                        ha="right" if dx < 0 else "left", fontsize=7, color=F.INK if g == "G5" else F.INK_2,
                        zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color=F.INK_3, lw=0.5, shrinkA=0, shrinkB=3) if far else None)
        ax.axhline(0, color=F.INK_3, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks[task]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        lo_x, hi_x = ax.get_xlim()
        ax.set_xlim(lo_x, hi_x * 1.6)
        ax.set_title(label)
        ax.grid(axis="y", color=F.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("step-0 ‖∂L/∂B‖ (log)")
    axes[0].set_ylabel("validation score − G1 (pp, 95% CI)")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(os.path.join(out, f"step0_vs_accuracy.{ext}"), facecolor="white", metadata=meta)
    plt.close(fig)


s, g = F.scores()
step0_vs_accuracy(s, g)
print("ok")
