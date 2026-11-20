"""Compare G8 with G3 seed by seed, and (optionally) G8's initial A with G3's.

    python jobs/summarize.py [--g3_cache PATH_TO_G3G4_cache]
Reads results/{main,lowres}/*_G8_*.json and, as references, G3's JSONs from ../G3G4/results (falling back to
the copies in ref/G3/ outside the repo) and G1's from ../G1G2/results when present.
With --g3_cache it also compares cache/g8A_* with G3's g3A_* (same task and seed). Writes results/g8_runs.csv
and results/g8_summary.txt.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alloc import subspace_overlap  # noqa: E402

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}   # two-sided t quantiles, df = n - 1

ap = argparse.ArgumentParser()
ap.add_argument("--g3_cache", default=None)
args = ap.parse_args()


def load(pattern):
    return {(r["task"], r["seed"]): r for r in (json.load(open(p)) for p in sorted(glob.glob(pattern)))}


def ref(group, folder, stage):
    """The group's JSONs from its folder in the repo, else from the copies in ref/<group>/."""
    return load(f"../{folder}/results/{stage}/{stage}_{group}_*.json") or load(f"ref/{group}/{stage}_{group}_*.json")


g8 = load("results/main/main_G8_*.json")
g3 = ref("G3", "G3G4", "main")
g1 = ref("G1", "G1G2", "main")
if not g8:
    sys.exit("no G8 results under results/main/")

out = open("results/g8_summary.txt", "w", encoding="utf-8", newline="\n")


def say(text=""):
    print(text)
    out.write(text + "\n")


with open("results/g8_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "task", "seed", "lr", "acc", "f1", "primary", "g3_primary", "diff_pp",
                "step0_grad_B", "lora_params", "init_s", "total_s", "peak_mem_gb", "gpu", "slurm_job"])
    for key, r in sorted(g8.items()):
        p3 = g3[key]["final"]["primary"] if key in g3 else ""
        w.writerow([r["run_id"], r["task"], r["seed"], r["lr"], r["final"]["acc"], r["final"].get("f1", ""),
                    r["final"]["primary"], p3, (r["final"]["primary"] - p3) * 100 if p3 != "" else "",
                    r["step0_grad_B"], r["lora_params"], r["time_s"]["init"], r["time_s"]["total"],
                    r["peak_mem_gb"], r["gpu"], r["slurm_job"]])

say("G8 vs G3, official validation, final epoch (MRPC: F1; SST-2 / RTE: accuracy), %")
say(f"{'task':6s} {'n':>2s}  {'G8 mean±sd':>13s}  {'G3 mean±sd':>13s}  {'G8 − G3 [95% CI]':>24s}")
for task in ["sst2", "mrpc", "rte"]:
    seeds = sorted(s for t, s in g8 if t == task and (t, s) in g3)
    if not seeds:
        continue
    a = np.array([g8[(task, s)]["final"]["primary"] for s in seeds]) * 100
    b = np.array([g3[(task, s)]["final"]["primary"] for s in seeds]) * 100
    d = a - b
    n = len(d)
    half = T95[n - 1] * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    say(f"{task:6s} {n:2d}  {a.mean():6.2f} ± {a.std(ddof=1):4.2f}  {b.mean():6.2f} ± {b.std(ddof=1):4.2f}"
        f"  {d.mean():+6.2f} [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}]")

lr8 = load("results/lowres/lowres_G8_*.json")


def paired(mine, other, task, label, note=""):
    seeds = sorted(s for t, s in mine if t == task and (t, s) in other)
    if not seeds:
        return
    a = np.array([mine[(task, s)]["final"]["primary"] for s in seeds]) * 100
    b = np.array([other[(task, s)]["final"]["primary"] for s in seeds]) * 100
    d = a - b
    n = len(d)
    half = T95[n - 1] * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    say(f"{label:6s} {n:2d}  {a.mean():6.2f} ± {a.std(ddof=1):4.2f}  {b.mean():6.2f} ± {b.std(ddof=1):4.2f}"
        f"  {d.mean():+6.2f} [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}]{note}")


paired(lr8, ref("G3", "G3G4", "lowres"), "sst2", "lowres", "   (SST-2 1k subset)")
if g1:
    say()
    say(f"G8 vs G1 (plain LoRA, its own chosen lr), same seeds")
    say(f"{'task':6s} {'n':>2s}  {'G8 mean±sd':>13s}  {'G1 mean±sd':>13s}  {'G8 − G1 [95% CI]':>24s}")
    for task in ["sst2", "mrpc", "rte"]:
        paired(g8, g1, task, task)
    paired(lr8, ref("G1", "G1G2", "lowres"), "sst2", "lowres", "   (SST-2 1k subset)")

say()
say("step-0 ||dL/dB|| (mean over seeds): G8 vs G3")
for task in ["sst2", "mrpc", "rte"]:
    seeds = sorted(s for t, s in g8 if t == task and (t, s) in g3)
    if seeds:
        a = np.mean([g8[(task, s)]["step0_grad_B"] for s in seeds])
        b = np.mean([g3[(task, s)]["step0_grad_B"] for s in seeds])
        say(f"{task:6s} {a:8.3f}  {b:8.3f}  ratio {a / b:.2f}")

if args.g3_cache:
    say()
    say("initial A: subspace overlap G8 vs G3 (mean over 72 layers) and rank-allocation L1 distance (of 576)")
    for key in sorted(g8):
        task, seed = key
        p8 = f"cache/g8A_{task}_False_s{seed}.pt"
        p3 = os.path.join(args.g3_cache, f"g3A_{task}_False_s{seed}.pt")
        if not (os.path.exists(p8) and os.path.exists(p3)):
            continue
        a8, a3 = torch.load(p8, weights_only=True), torch.load(p3, weights_only=True)
        names = sorted(set(a8) | set(a3))
        ov = [subspace_overlap(a8[n], a3[n]) for n in names if n in a8 and n in a3]
        l1 = sum(abs((a8[n].shape[0] if n in a8 else 0) - (a3[n].shape[0] if n in a3 else 0)) for n in names)
        say(f"{task:6s} s{seed}  overlap {np.nanmean(ov):.3f}  L1 {l1}")
out.close()
