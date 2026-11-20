"""Export every run the summary uses into summary/site/data.js for the results page (summary/site/index.html).

    python summary/site/build_data.py      # from the repository root

Reads the same JSONs as summary/collect.py: */results/{main,lowres}/ (shared protocol), */results/control*/ (controls),
*/results/seeds10/ (extra seeds), the H4 overlaps and the per-layer H4 rebuild, and the H1 diagnosis
("H1 Diagnosis/results/<task>_seed<k>.json"). The page computes paired differences
itself from the per-seed values, with the same t-based 95% intervals as collect.py.
"""
import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "summary", "site", "data.js")
SOURCES = {"main": "shared", "lowres": "shared", "control": "lr control", "control_g8": "lr control",
           "control_repro": "L40S rerun", "control_rho1": "uniform ranks", "seeds10": "extra seeds"}
BIN = 10   # loss curves: mean over bins of 10 steps (500 steps -> 50 points)


def short_gpu(g):
    if not g:
        return "not recorded"
    return g.replace("NVIDIA ", "").replace("GeForce ", "")


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "results", "*", "**", "*.json"), recursive=True)):
        parts = os.path.relpath(path, ROOT).split(os.sep)
        src = parts[2]
        if src not in SOURCES:
            continue
        r = json.load(open(path))
        if "group" not in r or "final" not in r or r.get("shared_protocol_compliant") is False:
            continue
        task = "sst2" if r["task"] == "sst2_lowres" else r["task"]
        setting = "lowres" if r["stage"] == "lowres" else task
        rho = (r.get("cfg") or {}).get("rho") if r["group"] in ("G3", "G4") else None
        kind = SOURCES[src]
        gpu = short_gpu(r.get("gpu"))
        loss = r.get("loss_first_steps")
        runs.append(dict(g=r["group"], s=setting, lr=r["lr"], seed=r["seed"], v=round(100 * r["final"]["primary"], 4),
                         acc=round(100 * r["final"]["acc"], 4), kind=kind, gpu=gpu, rho=rho,
                         step0=r.get("step0_grad_B"), params=r.get("lora_params"), folder=parts[0],
                         # training-batch loss of the first 500 LoRA steps, averaged over bins of BIN steps
                         loss=[round(sum(loss[i:i + BIN]) / BIN, 4) for i in range(0, 500, BIN)]
                         if loss and len(loss) >= 500 else None))

    # H4 subspace overlaps with same-task, same-seed G3 (summary level, as in collect.py)
    h4 = []
    g9 = os.path.join(ROOT, "G9", "results", "h4_overlap.csv")
    for r in csv.DictReader(open(g9)):
        h4.append(dict(kind="G9", task=r["task"], seed=int(r["seed"]), v=float(r["overlap"])))
    for line in open(os.path.join(ROOT, "G8", "results", "g8_summary.txt")):
        f = line.split()
        if len(f) >= 4 and f[2] == "overlap":
            h4.append(dict(kind="G8", task=f[0], seed=int(f[1][1:]), v=float(f[3])))
    for p in glob.glob(os.path.join(ROOT, "*", "results", "noise_floor", "noise_floor_*.json")):
        r = json.load(open(p))
        h4.append(dict(kind="floor", task=r["task"], seed=r["seed"], v=r["overlap_mean"]))
        h4.append(dict(kind="random", task=r["task"], seed=r["seed"], v=r["random_overlap_mean"]))

    # Per-layer rebuild on one GPU: mean over the 13 task-seed pairs for every (type, depth)
    acc = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "results", "h4_layers", "h4_*.json"))):
        r = json.load(open(p))
        for l in r["layers"].values():
            for k in ("floor", "G8", "G9"):
                acc[(l["type"], l["depth"])][k].append(l[f"{k}_vs_G3"])
    layers = [dict(type=t, depth=d, **{k: round(sum(v) / len(v), 4) for k, v in ks.items()})
              for (t, d), ks in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0]))]

    # H1: GW-EVA vs EVA on the same calibration sample, with a GW-vs-GW floor from disjoint samples
    h1, h1_acc = [], defaultdict(lambda: defaultdict(list))
    tname = {"FFN input": "ffn_in", "FFN output": "ffn_out"}
    for p in sorted(glob.glob(os.path.join(ROOT, "H1 Diagnosis", "results", "*_seed*.json"))):
        r = json.load(open(p))
        rows = r["per_layer"]
        pr = [l["pearson"] for l in rows if l["pearson"] is not None]
        h1.append(dict(task="lowres" if r["task"] == "sst2_lowres" else r["task"], seed=r["seed"],
                       eva_gw=round(r["mean_eva_gw"], 4), gw_gw=round(r["mean_gw_gw"], 4),
                       above=sum(l["eva_gw"] >= l["gw_gw"] for l in rows), layers=len(rows),
                       angle=round(sum(sum(l["principal_angles_degrees"]) / 8 for l in rows) / len(rows), 2),
                       pearson=round(sum(pr) / len(pr), 4) if pr else None))
        if r["task"] != "sst2_lowres":   # per-layer map: the three full tasks only
            for l in rows:
                for k in ("eva_gw", "gw_gw"):
                    h1_acc[(tname.get(l["layer_type"], l["layer_type"]), l["block"])][k].append(l[k])
    h1_layers = [dict(type=t, depth=d, **{k: round(sum(v) / len(v), 4) for k, v in ks.items()})
                 for (t, d), ks in sorted(h1_acc.items(), key=lambda kv: (kv[0][1], kv[0][0]))]

    data = dict(runs=runs, h4=h4, h4_layers=layers, h1=h1, h1_layers=h1_layers, bin=BIN)
    with open(OUT, "w") as f:
        f.write("// Generated by summary/site/build_data.py; do not edit by hand.\n")
        f.write("window.GWEVA_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"{len(runs)} runs, {sum(r['loss'] is not None for r in runs)} with loss curves, {len(h4)} H4 overlaps, "
          f"{len(layers)} layer cells, {len(h1)} H1 runs -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
