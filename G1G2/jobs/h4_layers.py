"""Per-layer H4 overlaps on one GPU: same-task EVA (G3) against a disjoint same-task sample, G8 and G9.

    python jobs/h4_layers.py --out results/h4_layers      # needs ../G3G4/src, ../G8/src and gweva-data/wikitext103

For each task and seed this rebuilds four initial lora_A sets with the groups' own calibration rules, all with G3's
EVA (PEFT, r = 8, rho = 2, label mask off):
  G3     the task's training split, perm[:1024] of the seed (data.calib_loader)
  floor  the same split, perm[1024:2048] (disjoint from G3)
  G8     G3's examples and token layouts with wikitext-103 text (G8/src/generic.py)
  G9     the source task's training split, perm[:1024] (G9's CROSS_CALIB_MAP)
and compares floor, G8 and G9 with G3 layer by layer (alloc.subspace_overlap). G8/G9's own caches are not in git, so
this makes their overlaps reproducible from the repository and adds the per-layer view.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import eva_noise_floor as nf   # puts ../G3G4/src on sys.path

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "G8", "src"))
import alloc  # noqa: E402  (G3G4's)
import data  # noqa: E402
import generic  # noqa: E402  (G8's; it imports the identical data module above)
import init_g34  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

SEEDS = {"sst2": [0, 1, 2], "mrpc": [0, 1, 2, 3, 4], "rte": [0, 1, 2, 3, 4]}
G9_SOURCE = {"sst2": "mrpc", "mrpc": "rte", "rte": "mrpc"}   # G9/src/train.py CROSS_CALIB_MAP


def eva_from_loader(seed, loader, dev):
    torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(os.path.join(data.DATA_ROOT, "roberta-base"),
                                                              num_labels=2).to(dev)
    model = init_g34.init_g3(base, loader, nf.R, nf.ALPHA, nf.RHO)
    return {n.replace(init_g34.PREFIX, "").replace(".lora_A.default.weight", ""): p.detach().float().cpu()
            for n, p in model.named_parameters() if "lora_A" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h4_layers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(data.DATA_ROOT, "roberta-base"))
    train = {t: data.load_splits(t, tok, nf.MAX_LEN)["train"] for t in SEEDS}
    for task, seeds in SEEDS.items():
        for seed in seeds:
            path = os.path.join(args.out, f"h4_{task}_s{seed}.json")
            if os.path.exists(path):
                print(f"skip {path}: done")
                continue
            t0 = time.time()
            perm = np.random.default_rng(seed).permutation(len(train[task]))
            src = G9_SOURCE[task]
            perm_src = np.random.default_rng(seed).permutation(len(train[src]))

            def loader(ds, idx):
                return data.make_loader(ds.select(idx), tok, nf.BATCH, shuffle=True, seed=seed, drop_labels=True)
            a = {"G3": eva_from_loader(seed, loader(train[task], perm[:nf.CALIB]), dev),
                 "floor": eva_from_loader(seed, loader(train[task], perm[nf.CALIB:2 * nf.CALIB]), dev),
                 "G8": eva_from_loader(seed, generic.generic_calib_loader(train[task], tok, nf.CALIB, nf.BATCH, seed),
                                       dev),
                 "G9": eva_from_loader(seed, loader(train[src], perm_src[:nf.CALIB]), dev)}
            layers = {}
            for name in sorted(a["G3"]):
                layers[name] = dict(type=nf.layer_type(name), depth=int(name.split("layer.")[1].split(".")[0]),
                                    ranks={k: int(v[name].shape[0]) for k, v in a.items()},
                                    **{f"{k}_vs_G3": alloc.subspace_overlap(a[k][name], a["G3"][name])
                                       for k in ("floor", "G8", "G9")})
            res = dict(task=task, seed=seed, g9_source=src,
                       mean={k: float(np.mean([v[f"{k}_vs_G3"] for v in layers.values()])) for k in ("floor", "G8", "G9")},
                       layers=layers, time_s=time.time() - t0,
                       gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu")
            with open(path + ".tmp", "w") as f:
                json.dump(res, f, indent=1)
            os.replace(path + ".tmp", path)
            print(f"done {task} s{seed}: " + ", ".join(f"{k} {v:.3f}" for k, v in res["mean"].items())
                  + f", {res['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
