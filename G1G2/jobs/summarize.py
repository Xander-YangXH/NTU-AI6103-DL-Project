"""Summarise G1/G2: tuning table, main results (mean ± sd), paired G2 − G1 differences, low-resource G1.

    python jobs/summarize.py        # writes results/g12_runs.csv and results/g12_summary.txt
"""
import csv
import glob
import json
import math
import os

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
TASKS = ["sst2", "mrpc", "rte"]
GROUPS = ["G1", "G2"]
LR_GRID = [2e-4, 5e-4, 1e-3]


def load(stage):
    return [json.load(open(p)) for p in sorted(glob.glob(f"results/{stage}/{stage}_*.json"))]


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan"))


def paired(a, b):
    """Mean and 95% CI (t, df = n−1) of b − a over the seeds both have, in percentage points."""
    seeds = sorted(set(a) & set(b))
    d = [100 * (b[s] - a[s]) for s in seeds]
    m, sd = mean_sd(d)
    h = T95[len(d) - 1] * sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return len(d), m, m - h, m + h


def main():
    out = []
    tune = load("tune")
    if tune:
        out += ["## Learning-rate selection (dev, seed 0, primary %)", "",
                "| group | task | " + " | ".join(f"{lr:g}" for lr in LR_GRID) + " |", "|---|---|---|---|---|"]
        for g in GROUPS:
            for t in TASKS:
                by_lr = {r["lr"]: r["final"]["primary"] for r in tune if r["group"] == g and r["task"] == t}
                if by_lr:
                    out.append(f"| {g} | {t} | " + " | ".join(f"{100 * by_lr[lr]:.2f}" if lr in by_lr else "–"
                                                              for lr in LR_GRID) + " |")
        out.append("")

    runs = load("main") + load("lowres")
    rows = []
    for r in runs:
        rows.append(dict(stage=r["stage"], group=r["group"], task=r["task"], lr=r["lr"], seed=r["seed"],
                         primary=r["final"]["primary"], acc=r["final"]["acc"], f1=r["final"].get("f1"),
                         step0_grad_B=r["step0_grad_B"], A_row_norm=r["A_row_norm"]["mean"],
                         lora_params=r["lora_params"], total_s=r["time_s"]["total"], init_s=r["time_s"]["init"],
                         peak_mem_gb=r["peak_mem_gb"], gpu=r["gpu"]))
    if rows:
        os.makedirs("results", exist_ok=True)
        with open("results/g12_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    score = {}  # (stage, group, task) -> {seed: primary}
    for r in rows:
        score.setdefault((r["stage"], r["group"], r["task"]), {})[r["seed"]] = r["primary"]

    if any(k[0] == "main" for k in score):
        out += ["## Main results (official validation, mean ± sd, %)", "",
                "| task | seeds | G1 | G2 | G2 − G1 paired (95% CI) |", "|---|---|---|---|---|"]
        for t in TASKS:
            a, b = score.get(("main", "G1", t), {}), score.get(("main", "G2", t), {})
            cell = lambda d: "–" if not d else "{:.2f} ± {:.2f}".format(*mean_sd([100 * v for v in d.values()]))
            diff = "–"
            if len(set(a) & set(b)) > 1:
                n, m, lo, hi = paired(a, b)
                diff = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            out.append(f"| {t} | {max(len(a), len(b))} | {cell(a)} | {cell(b)} | {diff} |")
        out += ["", "## Per-run diagnostics (main, means over seeds)", "",
                "| group | task | lr | ‖∂L/∂B‖ step 0 | A row norm | total s | peak GB |", "|---|---|---|---|---|---|---|"]
        for g in GROUPS:
            for t in TASKS:
                rs = [r for r in rows if r["stage"] == "main" and r["group"] == g and r["task"] == t]
                if rs:
                    out.append(f"| {g} | {t} | {rs[0]['lr']:g} | {mean_sd([r['step0_grad_B'] for r in rs])[0]:.4f} | "
                               f"{rs[0]['A_row_norm']:.3f} | {mean_sd([r['total_s'] for r in rs])[0]:.0f} | "
                               f"{max(r['peak_mem_gb'] for r in rs):.2f} |")
        out.append("")

    low = score.get(("lowres", "G1", "sst2"))
    if low:
        m, sd = mean_sd([100 * v for v in low.values()])
        out += ["## Low-resource SST-2 1k (G1, official validation acc %)", "",
                " / ".join(f"s{s} {100 * low[s]:.2f}" for s in sorted(low)) + f"  →  {m:.2f} ± {sd:.2f}", ""]

    text = "\n".join(out)
    print(text)
    with open("results/g12_summary.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
