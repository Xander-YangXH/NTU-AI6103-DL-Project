"""Collect every group's final runs into one CSV and the cross-group tables.

    python summary/collect.py        # run from the repository root of the summary branch

Reads */results/{main,lowres}/*.json, writes summary/all_runs.csv and summary/tables.md.
Control runs outside the shared protocol (*/results/control/<stage>/*.json) get their own table and stay out of the CSV,
and so do extra seeds beyond the shared ones (*/results/seeds10/<stage>/*.json).
Early training is compared through each run's loss_first_steps (training-batch loss of the first 500 LoRA steps),
only between groups that used the same learning rate.
The H4 subspace overlaps with same-task EVA (G3) come from G9/results/h4_overlap.csv (G9/src/overlap_analysis.ipynb)
and G8/results/g8_summary.txt (G8/jobs/summarize.py); both use alloc.subspace_overlap averaged over the 72 layers.
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
SHARED = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]


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

    # Controls outside the shared protocol: */results/control*/<stage>/*.json (control = other learning rates,
    # control_repro = a rerun of the shared setting on another GPU, control_rho1 = EVA with uniform ranks).
    ctl = defaultdict(dict)   # (group, stage, task, lr, rho, gpu) -> {seed: primary}
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "results", "control*", "*", "*.json"))):
        r = json.load(open(path))
        rho = (r.get("cfg") or {}).get("rho") if r["group"] in ("G3", "G4") else None
        ctl[(r["group"], r["stage"], r["task"], r["lr"], rho, r.get("gpu", "").replace("NVIDIA ", ""))][r["seed"]] = \
            r["final"]["primary"]
    if ctl:
        def diff(a, b):
            if len(set(a) & set(b)) < 2:
                return "–"
            n, dm, lo, hi = paired(a, b)
            return f"{dm:+.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else "")

        # G1 at each learning rate it has been run at (shared run plus controls), as the reference for the rows below
        g1 = {lr: d for (g, st, t, lr, _, _), d in ctl.items() if g == "G1" and st == "lowres"}
        g1[meta[("G1", "lowres", "sst2")][0]["lr"]] = score[("G1", "lowres", "sst2")]
        out += ["", "## Controls outside the shared protocol", "",
                "Low-resource SST-2 1k. Shared rows repeat the shared-protocol runs for reference; ρ is EVA's rank "
                "redistribution factor (2 = original, 1 = uniform r = 8). The \"shared setting\" column compares with "
                "the same group's shared-protocol run on the same GPU when one exists (G3 on L40S: the lr 1e-3, ρ 2 "
                "rerun), so the G3 rows isolate learning rate and ρ from the GPU change.", "",
                "| Run | lr | ρ | GPU | Result | − same group's shared setting | − G1 at the same lr | − G1 at 5e-4 |",
                "|---|---|---|---|---|---|---|---|"]
        rows = [(("G1", "lowres", "sst2", meta[("G1", "lowres", "sst2")][0]["lr"], None, "L40S"),
                 score[("G1", "lowres", "sst2")], "G1 (shared)"),
                (("G3", "lowres", "sst2", meta[("G3", "lowres", "sst2")][0]["lr"], 2.0, "A40"),
                 score[("G3", "lowres", "sst2")], "G3 (shared)")]
        rows += [(k, d, k[0]) for k, d in sorted(ctl.items(), key=lambda kv: (kv[0][0], -kv[0][3], -(kv[0][4] or 0)))]
        for (g, st, t, lr, rho, gpu), d, label in rows:
            m, sd = mean_sd([100 * v for v in d.values()])
            # same group's shared setting, rerun on the same GPU when there is such a rerun (G3: control_repro)
            shared_lr = meta[(g, st, t)][0]["lr"]
            shared_rho = (meta[(g, st, t)][0].get("rho") or 2.0) if rho is not None else None   # EVA's default rho = 2
            shared = ctl.get((g, st, t, shared_lr, shared_rho, gpu), score.get((g, st, t), {}))
            if shared is d:
                shared = score.get((g, st, t), {})
            out.append(f"| {label} | {lr:g} | {'–' if rho is None else f'{rho:g}'} | {gpu} | {m:.2f} ± {sd:.2f} | "
                       f"{'–' if 'shared' in label else diff(shared, d)} | "
                       f"{'–' if g == 'G1' else diff(g1.get(lr, {}), d)} | "
                       f"{'–' if g == 'G1' else diff(g1.get(5e-4, {}), d)} |")

    # Early training: mean training loss over step windows, paired by seed, only at equal learning rates.
    # The low-resource G1 reference is the lr 1e-3 control, which matches G3/G5's learning rate there.
    curves = defaultdict(dict)   # (group, setting) -> {seed: (lr, losses)}
    paths = (glob.glob(os.path.join(ROOT, "*", "results", "main", "*.json"))
             + glob.glob(os.path.join(ROOT, "*", "results", "lowres", "*.json"))
             + glob.glob(os.path.join(ROOT, "*", "results", "control", "lowres", "*.json")))
    for path in paths:
        r = json.load(open(path))
        if "group" not in r or not r.get("loss_first_steps") or r.get("shared_protocol_compliant") is False:
            continue
        ctl = os.sep + "control" + os.sep in path
        if ctl and r["group"] != "G1":   # only the G1 lr control is used as a reference here
            continue
        setting = "low-res" if r["stage"] == "lowres" else r["task"]
        curves[(r["group"] + ("@ctl" if ctl else ""), setting)][r["seed"]] = (r["lr"], r["loss_first_steps"])
    windows = [("1–50", 0, 50), ("51–200", 50, 200), ("201–500", 200, 500)]

    def window_diff(a, b, lo, hi):
        seeds = sorted(set(a) & set(b))
        d = [sum(b[s][1][lo:hi]) / (hi - lo) - sum(a[s][1][lo:hi]) / (hi - lo) for s in seeds]
        m, sd = mean_sd(d)
        h = T95[len(d) - 1] * sd / math.sqrt(len(d))
        return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}]" + (" *" if m - h > 0 or m + h < 0 else "")

    out += ["", "## Early training: paired difference in training loss (b − a), equal learning rates only", "",
            "Mean training-batch loss over each step window; negative = b fits faster. G1@ctl is G1 at lr 1e-3 "
            "(the low-resource control). Batches are in the same order for G1/G2/G3/G4 (shared loader) but not "
            "for G5/G6, which use their own loader.", "",
            "| b − a | Setting | lr | Seeds | " + " | ".join(f"Steps {n}" for n, _, _ in windows) + " |",
            "|---|---|---|---|" + "---|" * len(windows)]
    pairs = [("G3", "G5"), ("G3", "G8"), ("G3", "G9"), ("G3", "G4"), ("G5", "G6"), ("G1", "G2"), ("G1", "G3"), ("G1", "G5"),
             ("G1", "G7"), ("G1@ctl", "G3"), ("G1@ctl", "G5")]
    for ga, gb in pairs:
        for setting in ("sst2", "mrpc", "rte", "low-res"):
            a, b = curves.get((ga, setting), {}), curves.get((gb, setting), {})
            seeds = sorted(set(a) & set(b))
            if len(seeds) < 2 or a[seeds[0]][0] != b[seeds[0]][0]:
                continue
            out.append(f"| {gb} − {ga} | {setting} | {a[seeds[0]][0]:g} | {len(seeds)} | "
                       + " | ".join(window_diff(a, b, lo, hi) for _, lo, hi in windows) + " |")
    out += ["", "`*` the interval excludes 0."]

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

    # H4: overlap of each group's initial lora_A with same-task, same-seed G3, per target task.
    # G9 on task t is EVA calibrated on another task (src below); G8 is EVA calibrated on wikitext-103 laid out
    # like G3's calibration batches.
    h4 = defaultdict(lambda: defaultdict(dict))   # group -> task -> {seed: overlap}
    g9_path = os.path.join(ROOT, "G9", "results", "h4_overlap.csv")
    if os.path.exists(g9_path):
        for r in csv.DictReader(open(g9_path)):
            h4["G9"][r["task"]][int(r["seed"])] = float(r["overlap"])
    g8_path = os.path.join(ROOT, "G8", "results", "g8_summary.txt")
    if os.path.exists(g8_path):
        for line in open(g8_path):
            f = line.split()   # e.g. "mrpc   s0  overlap 0.736  L1 54"
            if len(f) >= 4 and f[2] == "overlap":
                h4["G8"][f[0]][int(f[1][1:])] = float(f[3])
    if h4:
        src = {"G8": {t: "wikitext-103" for t in ("sst2", "mrpc", "rte")},
               "G9": {"sst2": "mrpc", "mrpc": "rte", "rte": "mrpc"}}
        out += ["", "## H4: subspace overlap of EVA calibrated elsewhere with same-task EVA (G3)", "",
                "Mean squared cosine of the principal angles between the two lora_A row spaces, averaged over the "
                "72 layers; 1 = identical, at most about 0.02 for random subspaces of these ranks (r ≤ 16 in d ≥ 768).", "",
                "| Group | Target task | Calibrated on | Seeds | Overlap (mean ± sd) | Range | Accuracy − G3 |",
                "|---|---|---|---|---|---|---|"]
        for g in sorted(h4):
            for t in ("sst2", "mrpc", "rte"):
                d = h4[g].get(t)
                if not d:
                    continue
                m, sd = mean_sd(list(d.values()))
                n, dm, lo, hi = paired(score.get(("G3", "main", t), {}), score.get((g, "main", t), {}))
                out.append(f"| {g} | {t} | {src[g][t]} | {len(d)} | {m:.3f} ± {sd:.3f} | "
                           f"{min(d.values()):.3f}–{max(d.values()):.3f} | {dm:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                           + (" *" if lo > 0 or hi < 0 else "") + " |")
        if h4["G9"].get("mrpc") and h4["G9"].get("rte"):
            same = max(abs(h4["G9"]["mrpc"][s] - h4["G9"]["rte"][s]) for s in h4["G9"]["mrpc"])
            out += ["", f"G9's MRPC and RTE rows compare the same two subspaces (EVA on MRPC vs EVA on RTE) and agree "
                        f"to {same:.0e} per seed, so G9 gives two independent pairs, not three."]

    text = "\n".join(out) + "\n"
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
