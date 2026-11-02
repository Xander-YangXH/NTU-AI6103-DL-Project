"""Collect G3/G4 results into CSVs for the paired statistics, the rank heatmaps and the report.

    python jobs/summarize.py           # run on the cluster inside a CPU-only job, or locally after download
Writes results/g34_runs.csv, results/g34_ranks.csv and prints the summary tables (also saved to results/g34_summary.txt).
Then `python jobs/plot_ranks.py` draws the per-layer rank heatmaps from g34_ranks.csv.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from alloc import subspace_overlap  # noqa: E402
from make_queue import pick_best  # noqa: E402  (same tie rule as the main-stage queue)

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}   # two-sided t quantiles, df = n - 1
KINDS = ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]


def kind(layer):
    for k in KINDS:
        if layer.endswith(k) and not (k == "output.dense" and "attention" in layer):
            return k
    return "?"


runs = [json.load(open(p)) for p in sorted(glob.glob("results/*/*.json"))]
if not runs:
    sys.exit("no results found under results/")


class _Tee:
    """Print to the console and to results/g34_summary.txt (UTF-8, LF) at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.stdout, open("results/g34_summary.txt", "w", encoding="utf-8", newline="\n"))

with open("results/g34_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "stage", "group", "task", "lr", "seed", "eval_split", "acc", "f1", "primary",
                "step0_grad_B", "lora_params", "budget_dev", "init_s", "total_s", "peak_mem_gb", "slurm_job"])
    for r in runs:
        w.writerow([r["run_id"], r["stage"], r["group"], r["task"], r["lr"], r["seed"], r["eval_split"],
                    r["final"]["acc"], r["final"].get("f1", ""), r["final"]["primary"], r["step0_grad_B"],
                    r["lora_params"], r["budget_dev"], r["time_s"]["init"], r["time_s"]["total"],
                    r["peak_mem_gb"], r["slurm_job"]])
with open("results/g34_ranks.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "group", "task", "seed", "layer", "kind", "rank"])
    for r in runs:
        for layer, k in r["ranks"].items():
            w.writerow([r["run_id"], r["group"], r["task"], r["seed"], layer, kind(layer), k])

print("== tuning: best learning rate on dev (primary metric; send G3's to the G8/G9 owner) ==")
tune = defaultdict(list)
for r in runs:
    if r["stage"] == "tune":
        tune[(r["group"], r["task"])].append(r)
for (g, t), rs in sorted(tune.items()):
    best = pick_best(rs)
    grid = "  ".join(f"{r['lr']:g}:{r['final']['primary'] * 100:.1f}" for r in sorted(rs, key=lambda r: r["lr"]))
    print(f"{g} {t:5s} best lr={best['lr']:g}   ({grid})")
print()

main = [r for r in runs if r["stage"] == "main"]
print("== main: official validation, mean ± sd (primary metric: acc, MRPC F1) ==")
by = defaultdict(dict)
for r in main:
    by[(r["group"], r["task"])][r["seed"]] = r
for (g, t), d in sorted(by.items()):
    v = np.array([x["final"]["primary"] for x in d.values()]) * 100
    p = np.array([x["lora_params"] for x in d.values()])
    dev = np.array([x["budget_dev"] for x in d.values()])
    print(f"{g} {t:5s} n={len(v)}  {v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}   "
          f"LoRA params {p.mean():,.0f} ({dev.mean():+.1%} vs budget)")

print("\n== paired difference G4 − G3 on the same seeds, 95% CI ==")
for t in ("sst2", "mrpc", "rte"):
    common = sorted(set(by.get(("G3", t), {})) & set(by.get(("G4", t), {})))
    if len(common) < 2:
        continue
    diff = np.array([by[("G4", t)][s]["final"]["primary"] - by[("G3", t)][s]["final"]["primary"] for s in common]) * 100
    half = T95[len(diff) - 1] * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"{t:5s} n={len(diff)}  {diff.mean():+.2f}  [{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}]")

print("\n== mean rank by layer type (all G3/G4 runs) ==")
agg = defaultdict(list)
for r in runs:
    for layer, k in r["ranks"].items():
        agg[(r["group"], kind(layer))].append(k)
for g in ("G3", "G4"):
    print(g, "  ".join(f"{k}={np.mean(agg[(g, k)]):.1f}" for k in KINDS if agg[(g, k)]))

print("\n== check that G3 and G4 differ only in allocation ==")
for a_path in sorted(glob.glob("cache/g3A_*.pt")):
    c_path = a_path.replace("g3A_", "eva16_")
    if not os.path.exists(c_path):
        continue
    a, c = torch.load(a_path), torch.load(c_path)
    ov = [subspace_overlap(a[n], c["comps"][n][: a[n].shape[0]]) for n in a if n in c["comps"]]
    tag = os.path.basename(a_path)[4:-3]
    g3 = next((r for r in runs if r["group"] == "G3" and f"{r['task']}_{r['stage'] == 'lowres'}_s{r['seed']}" == tag), None)
    g4 = next((r for r in runs if r["group"] == "G4" and f"{r['task']}_{r['stage'] == 'lowres'}_s{r['seed']}" == tag), None)
    l1 = sum(abs(g3["ranks"].get(k, 0) - v) for k, v in g4["g3_rule_on_cache"].items()) if g3 and g4 else None
    total = sum(g3["ranks"].values()) if g3 else "?"
    print(f"{tag}: mean subspace overlap G3 vs cache = {np.nanmean(ov):.3f} (min {np.nanmin(ov):.3f}); "
          f"L1 between PEFT's G3 ranks and EVA's rule on the cache = {l1} of {total} ranks")
