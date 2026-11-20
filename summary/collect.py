"""Collect every group's final runs into one CSV and the cross-group tables.

    python summary/collect.py        # run from the repository root of the summary branch

Reads */results/{main,lowres}/*.json, writes summary/all_runs.csv and summary/tables.md.
Control runs outside the shared protocol (*/results/control/<stage>/*.json) get their own table and stay out of the CSV,
and so do extra seeds beyond the shared ones (*/results/seeds10/<stage>/*.json).
G9's H4 measurement (G9/results/h4_overlap.csv, written by G9/src/overlap_analysis.ipynb) is summarised per task pair.
All groups share the data split, so they are paired by seed. G5/G6 name their low-resource task "sst2_lowres";
it is read as task "sst2" of stage "lowres" like everyone else's. Runs that still carry
shared_protocol_compliant = False are skipped.
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
SHARED = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9"]


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
        if stage == "pilot" or r.get("shared_protocol_compliant") is False:   # pilots and off-protocol runs
            continue
        if r["task"] == "sst2_lowres":
            r["task"] = "sst2"
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
    out += ["", "Seeds: SST-2 3, MRPC / RTE 5, low-resource 5 (G5/G6: 3; paired on their common seeds). Learning rates:", "",
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
    for g in SHARED:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        gpus = ", ".join(sorted({r["gpu"] or "not recorded" for r in rs}))
        vers = ", ".join(sorted({f"{r['torch']} / {r['transformers']} / {r['peft']}" if r["torch"] else "not recorded"
                                 for r in rs}))
        p = sorted({r["lora_params"] for r in rs})
        params = f"{p[0]:,}" if len(p) == 1 else f"{p[0]:,}–{p[-1]:,}"
        out.append(f"| {g} | {len(rs)} | {gpus} | {vers} | {params} |")

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

    extra = defaultdict(dict)   # (group, stage, task) -> {seed: primary}, seeds beyond the shared ones
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "results", "seeds10", "*", "*.json"))):
        r = json.load(open(path))
        extra[(r["group"], r["stage"], r["task"])][r["seed"]] = r["final"]["primary"]
    if extra:
        out += ["", "## Extra seeds (outside the shared tables until every group has them)", "",
                "| Run | lr | Seeds | Shared seeds | All seeds | Single-class runs |", "|---|---|---|---|---|---|"]
        for (g, stage, task), d in sorted(extra.items()):
            base = score.get((g, stage, task), {})
            both = {**base, **d}
            m0, sd0 = mean_sd([100 * v for v in base.values()])
            m, sd = mean_sd([100 * v for v in both.values()])
            lr = meta[(g, stage, task)][0]["lr"]
            # a run that predicts one class scores exactly one class's share (RTE val: 146 / 277 or 131 / 277)
            collapsed = sorted(s for s, v in both.items()
                               if task == "rte" and min(abs(v - 146 / 277), abs(v - 131 / 277)) < 1e-9)
            out.append(f"| {g} {stage} {task} | {lr:g} | {len(both)} | {m0:.2f} ± {sd0:.2f} | {m:.2f} ± {sd:.2f} | "
                       + (", ".join(f"seed {s}" for s in collapsed) if collapsed else "–") + " |")

    h4_path = os.path.join(ROOT, "G9", "results", "h4_overlap.csv")
    if os.path.exists(h4_path):
        # G9 on task t is EVA calibrated on SRC[t], so overlap(G9_t, G3_t) compares EVA(SRC[t]) with EVA(t) at the
        # same seed. MRPC and RTE borrow from each other: their rows are the same pair measured twice.
        src = {"sst2": "mrpc", "mrpc": "rte", "rte": "mrpc"}
        h4 = defaultdict(dict)
        for r in csv.DictReader(open(h4_path)):
            h4[r["task"]][int(r["seed"])] = float(r["overlap"])
        out += ["", "## H4: subspace overlap of cross-task EVA (G9) with same-task EVA (G3)", "",
                "Mean squared cosine of the principal angles between the two lora_A row spaces, averaged over the "
                "72 layers; 1 = identical, at most about 0.02 for random subspaces of these ranks (r ≤ 16 in d ≥ 768).", "",
                "| Target task | Calibrated on | Seeds | Overlap (mean ± sd) | Range | G9 − G3 accuracy |",
                "|---|---|---|---|---|---|"]
        for t, d in h4.items():
            m, sd = mean_sd(list(d.values()))
            n, dm, lo, hi = paired(score.get(("G3", "main", t), {}), score.get(("G9", "main", t), {}))
            out.append(f"| {t} | {src[t]} | {len(d)} | {m:.3f} ± {sd:.3f} | {min(d.values()):.3f}–{max(d.values()):.3f} "
                       f"| {dm:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else "") + " |")
        same = max(abs(h4["mrpc"][s] - h4["rte"][s]) for s in h4["mrpc"])
        out += ["", f"The MRPC and RTE rows compare the same two subspaces (EVA on MRPC vs EVA on RTE) and agree to "
                    f"{same:.0e} per seed, so there are two independent pairs here, not three."]

    text = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
