"""Noise floor for the H4 subspace overlaps: EVA on one task, calibrated on two disjoint samples of its training split.

    python jobs/eva_noise_floor.py --out results/noise_floor      # all tasks, seeds 0-2; needs ../G3G4/src

Sample A is exactly G3's calibration set (the first 1,024 examples of the seed's permutation, as data.calib_loader);
sample B is the next 1,024. Each sample runs G3's EVA initialisation (PEFT, rho = 2) on a fresh RoBERTa-base, and the
two initial lora_A matrices are compared layer by layer with alloc.subspace_overlap, the function G8 and G9 used.
G8/G9's overlaps with G3 only mean "the subspace changed" if they are clearly below this same-task overlap.
Only the encoder's activations enter EVA, so the classification head (and its warm-up) does not matter here.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "G3G4", "src"))
import alloc  # noqa: E402  (G3G4's modules)
import data  # noqa: E402
import init_g34  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

R, ALPHA, RHO, CALIB, BATCH, MAX_LEN = 8, 8, 2.0, 1024, 16, 128
LAYER_TYPES = {"query": "q", "key": "k", "value": "v", "attention.output.dense": "o",
               "intermediate.dense": "ffn_in", "output.dense": "ffn_out"}


def layer_type(name):
    for key, short in LAYER_TYPES.items():   # attention.output.dense must win over output.dense
        if name.endswith(key):
            return short
    raise ValueError(name)


def eva_a(task, seed, idx, train_ds, tok, dev):
    """Initial lora_A of G3's EVA calibrated on train_ds[idx], keyed by target module name."""
    torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(os.path.join(data.DATA_ROOT, "roberta-base"),
                                                              num_labels=2).to(dev)
    loader = data.make_loader(train_ds.select(idx), tok, BATCH, shuffle=True, seed=seed, drop_labels=True)
    model = init_g34.init_g3(base, loader, R, ALPHA, RHO)
    return {n.replace(init_g34.PREFIX, "").replace(".lora_A.default.weight", ""): p.detach().float().cpu()
            for n, p in model.named_parameters() if "lora_A" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "rte"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/noise_floor")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(data.DATA_ROOT, "roberta-base"))
    for task in args.tasks:
        train_ds = data.load_splits(task, tok, MAX_LEN)["train"]
        for seed in args.seeds:
            path = os.path.join(args.out, f"noise_floor_{task}_s{seed}.json")
            if os.path.exists(path):
                print(f"skip {path}: done")
                continue
            t0 = time.time()
            perm = np.random.default_rng(seed).permutation(len(train_ds))   # as data.calib_loader
            assert len(perm) >= 2 * CALIB, (task, len(perm))
            a = eva_a(task, seed, perm[:CALIB], train_ds, tok, dev)
            b = eva_a(task, seed, perm[CALIB:2 * CALIB], train_ds, tok, dev)
            gen = torch.Generator().manual_seed(seed)
            layers = {}
            for name in sorted(a):
                ra, rb = a[name].shape[0], b[name].shape[0]
                rand = torch.linalg.qr(torch.randn(a[name].shape[1], rb, generator=gen, dtype=torch.float64))[0].T
                layers[name] = dict(type=layer_type(name), depth=int(re.search(r"layer\.(\d+)\.", name).group(1)),
                                    rank_a=ra, rank_b=rb, overlap=alloc.subspace_overlap(a[name], b[name]),
                                    random_overlap=alloc.subspace_overlap(a[name], rand))
            by_type = {t: float(np.mean([v["overlap"] for v in layers.values() if v["type"] == t]))
                       for t in LAYER_TYPES.values()}
            res = dict(task=task, seed=seed, calib_size=CALIB, sample_a="G3's calibration set (perm[:1024])",
                       sample_b="perm[1024:2048], disjoint from A",
                       overlap_mean=float(np.mean([v["overlap"] for v in layers.values()])),
                       random_overlap_mean=float(np.mean([v["random_overlap"] for v in layers.values()])),
                       rank_l1=int(sum(abs(v["rank_a"] - v["rank_b"]) for v in layers.values())),
                       by_type=by_type, layers=layers, time_s=time.time() - t0,
                       gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
                       versions={m.__name__: m.__version__ for m in (torch, __import__("transformers"),
                                                                       __import__("peft"))})
            with open(path + ".tmp", "w") as f:
                json.dump(res, f, indent=1)
            os.replace(path + ".tmp", path)
            print(f"done {task} s{seed}: overlap {res['overlap_mean']:.3f} (random {res['random_overlap_mean']:.3f}), "
                  f"rank L1 {res['rank_l1']}, {res['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
