"""Collect every group's final runs into one CSV and the cross-group tables.

    python summary/collect.py        # run from the repository root of the summary branch

Reads */results/{main,lowres}/*.json, writes summary/all_runs.csv and summary/tables.md.
Control runs outside the shared protocol (*/results/control/<stage>/*.json) get their own table and stay out of the CSV.
Groups on the shared protocol are paired by seed; G5/G6 follow an independent protocol
(shared_protocol_compliant = False in their JSONs) and are reported separately, never paired.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "summary")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
TASKS = [("main", "sst2", "SST-2 (acc)"), ("main", "mrpc", "MRPC (F1)"), ("main", "rte", "RTE (acc)"),
         ("lowres", "sst2", "Low-res SST-2 1k (acc)")]
SHARED = ["G1", "G2", "G3", "G4", "G7", "G9"]
INDEPENDENT = ["G5", "G6"]


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan"))


def paired(a, b):
    """b − a over common seeds, percentage points: (n, mean, lo, hi) with a t-based 95% CI."""
    seeds = sorted(set(a) & set(b))
    d = [100 * (b[s] - a[s]) for s in seeds]
    m, sd = mean_sd(d)
    h = T95[len(d) - 1] * sd / math.sqrt(len(d))
    return len(d), m, m - h, m + h


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "results", "*", "*.json"))):
        stage = os.path.basename(os.path.dirname(path))
        if stage not in ("main", "lowres", "pilot"):
            continue
        r = json.load(open(path))
        if "group" not in r or "final" not in r:
            continue
        if stage == "pilot":   # pilots are never part of the final tables
            continue
        rows.append(dict(
            folder=os.path.relpath(path, ROOT).split(os.sep)[0], stage=r["stage"], group=r["group"], task=r["task"],
            lr=r["lr"], seed=r["seed"], eval_split=r["eval_split"], primary=r["final"]["primary"],
            acc=r["final"]["acc"], f1=r["final"].get("f1"), step0_grad_B=r.get("step0_grad_B"),
            lora_params=r.get("lora_params"), budget_dev=r.get("budget_dev"),
            shared_protocol=r.get("shared_protocol_compliant", True), gpu=r.get("gpu"),
            torch=(r.get("versions") or {}).get("torch"), transformers=(r.get("versions") or {}).get("transformers"),
            peft=(r.get("versions") or {}).get("peft"), total_s=(r.get("time_s") or {}).get("total"),
        ))
    return rows


def main():
    rows = load()
    with open(os.path.join(OUT, "all_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    score = defaultdict(dict)   # (group, stage, task) -> {seed: primary}
    meta = defaultdict(list)
    for r in rows:
        score[(r["group"], r["stage"], r["task"])][r["seed"]] = r["primary"]
        meta[(r["group"], r["stage"], r["task"])].append(r)

    def cell(g, stage, task):
        d = score.get((g, stage, task))
        if not d:
            return "–"
        m, sd = mean_sd([100 * v for v in d.values()])
        return f"{m:.2f} ± {sd:.2f}"

    out = ["## Shared protocol: official validation, mean ± sd (%)", "",
           "| Task | " + " | ".join(SHARED) + " |", "|---|" + "---|" * len(SHARED)]
    for stage, task, label in TASKS:
        out.append(f"| {label} | " + " | ".join(cell(g, stage, task) for g in SHARED) + " |")
    out += ["", "Seeds: SST-2 3, MRPC / RTE / low-resource 5. Learning rates:", "",
            "| Task | " + " | ".join(SHARED) + " |", "|---|" + "---|" * len(SHARED)]
    for stage, task, label in TASKS:
        out.append(f"| {label} | " + " | ".join(
            (f"{meta[(g, stage, task)][0]['lr']:g}" if meta.get((g, stage, task)) else "–") for g in SHARED) + " |")

    for ref in ("G1", "G3"):
        others = [g for g in SHARED if g != ref]
        out += ["", f"## Paired difference vs {ref} (same seeds, percentage points, 95% CI)", "",
                "| Task | " + " | ".join(f"{g} − {ref}" for g in others) + " |", "|---|" + "---|" * len(others)]
        for stage, task, label in TASKS:
            cells = []
            for g in others:
                a, b = score.get((ref, stage, task), {}), score.get((g, stage, task), {})
                if len(set(a) & set(b)) > 1:
                    n, m, lo, hi = paired(a, b)
                    sig = " *" if lo > 0 or hi < 0 else ""
                    cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}")
                else:
                    cells.append("–")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("`*` the interval excludes 0.")

    out += ["", "## Step-0 ‖∂L/∂B‖ (mean over seeds)", "",
            "| Task | " + " | ".join(SHARED) + " |", "|---|" + "---|" * len(SHARED)]
    for stage, task, label in TASKS:
        cells = []
        for g in SHARED:
            v = [r["step0_grad_B"] for r in meta.get((g, stage, task), []) if r["step0_grad_B"] is not None]
            cells.append(f"{mean_sd(v)[0]:.2f}" if v else "–")
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    out += ["", "## Environment per group", "", "| Group | Runs | GPU | torch / transformers / peft | LoRA params |",
            "|---|---|---|---|---|"]
    for g in SHARED + INDEPENDENT:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        gpus = ", ".join(sorted({r["gpu"] for r in rs}))
        vers = ", ".join(sorted({f"{r['torch']} / {r['transformers']} / {r['peft']}" for r in rs}))
        p = sorted({r["lora_params"] for r in rs})
        params = f"{p[0]:,}" if len(p) == 1 else f"{p[0]:,}–{p[-1]:,}"
        out.append(f"| {g} | {len(rs)} | {gpus} | {vers} | {params} |")

    out += ["", "## G5 / G6: independent protocol (not pairable with the table above)", "",
            "| Task | G5 | G6 | G5 lr | G6 lr |", "|---|---|---|---|---|"]
    for stage, task, label in TASKS[:3]:
        lr = lambda g: (f"{meta[(g, stage, task)][0]['lr']:g}" if meta.get((g, stage, task)) else "–")
        out.append(f"| {label} | {cell('G5', stage, task)} | {cell('G6', stage, task)} | {lr('G5')} | {lr('G6')} |")

    ctl = defaultdict(dict)   # (group, task, lr) -> {seed: primary}
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "results", "control", "*", "*.json"))):
        r = json.load(open(path))
        ctl[(r["group"], r["stage"], r["task"], r["lr"])][r["seed"]] = r["final"]["primary"]
    if ctl:
        out += ["", "## Learning-rate controls (outside the shared protocol)", "",
                "| Run | lr | Result | This − shared-lr run | G3 − this |", "|---|---|---|---|---|"]
        for (g, stage, task, lr), d in sorted(ctl.items()):
            m, sd = mean_sd([100 * v for v in d.values()])
            diffs = []
            for a, b in ((score.get((g, stage, task), {}), d), (d, score.get(("G3", stage, task), {}))):
                n, dm, lo, hi = paired(a, b)
                diffs.append(f"{dm:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else ""))
            out.append(f"| {g} {stage} {task} | {lr:g} | {m:.2f} ± {sd:.2f} | {diffs[0]} | {diffs[1]} |")

    text = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
