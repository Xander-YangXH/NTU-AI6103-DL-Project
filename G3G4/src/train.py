"""One G3/G4 run: head warm-up -> EVA initialisation -> LoRA fine-tuning -> evaluation -> results JSON.

Usage: python src/train.py --stage {pilot,tune,main,lowres} --task {sst2,mrpc,rte} --group {G3,G4} --lr LR --seed S
A finished run is skipped, so a queue can be resubmitted after a timeout.
If member A's shared pipeline is ready, keep init_g34.py and replace the training loop below with A's.
"""
import argparse
import json
import os
import socket
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import alloc
import data
import init_g34

# ---- Shared settings: must match the other groups (agree with A and E before the tuning stage) ----
CFG = dict(
    model=os.path.join(data.DATA_ROOT, "roberta-base"), n_targets=72, r=8, alpha=8, rho=2.0, param_budget=1_327_104,
    max_len=128, batch_size=32, warmup_ratio=0.06, weight_decay=0.0,
    epochs={"sst2": 3, "mrpc": 10, "rte": 10, "lowres": 20},
    head_warmup_epochs=1, head_lr=1e-3,
    calib_size=1024, calib_batch_size=16, num_workers=2, loss_curve_steps=500,
)


def metrics(task, preds, labels):
    acc = float((preds == labels).mean())
    if task != "mrpc":
        return {"acc": acc, "primary": acc}
    tp = float(((preds == 1) & (labels == 1)).sum())
    f1 = 2 * tp / max(float((preds == 1).sum() + (labels == 1).sum()), 1.0)
    return {"acc": acc, "f1": f1, "primary": f1}


@torch.no_grad()
def evaluate(model, loader, task, dev):
    model.eval()
    preds, labels = [], []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits
        preds.append(logits.argmax(-1).cpu())
        labels.append(b["labels"].cpu())
    model.train()
    return metrics(task, torch.cat(preds).numpy(), torch.cat(labels).numpy())


def warm_up_head(model, loader, epochs, lr, dev):
    """Shared linear-probe warm-up: encoder frozen, only the classification head is trained."""
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("classifier")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for _ in range(epochs):
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(**b).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "tune", "main", "lowres"])
    ap.add_argument("--task", required=True, choices=list(data.TASK_KEYS))
    ap.add_argument("--group", required=True, choices=["G3", "G4"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    run_id = f"{args.stage}_{args.group}_{args.task}_lr{args.lr:g}_s{args.seed}"
    out_path = os.path.join(args.out, args.stage, run_id + ".json")
    if os.path.exists(out_path):
        print(f"skip {run_id}: done")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(CFG["model"])
    sp = data.load_splits(args.task, tok, CFG["max_len"])
    train_ds = sp["lowres"] if args.stage == "lowres" else sp["train"]
    eval_ds = sp["dev"] if args.stage in ("pilot", "tune") else sp["val"]
    nw = CFG["num_workers"]
    train_dl = data.make_loader(train_ds, tok, CFG["batch_size"], True, args.seed, nw)
    eval_dl = data.make_loader(eval_ds, tok, 128, False, args.seed, nw)
    calib = data.calib_loader(train_ds, tok, CFG["calib_size"], CFG["calib_batch_size"], args.seed, nw)

    base = AutoModelForSequenceClassification.from_pretrained(CFG["model"], num_labels=2).to(dev)
    targets = init_g34.target_names(base)
    assert len(targets) == CFG["n_targets"], len(targets)
    assert alloc.uniform_budget(base, targets, CFG["r"]) == CFG["param_budget"]

    t1 = time.time()
    warm_up_head(base, train_dl, CFG["head_warmup_epochs"], CFG["head_lr"], dev)
    t2 = time.time()

    extra = {}
    cache_path = os.path.join(args.cache, f"eva16_{args.task}_{args.stage == 'lowres'}_s{args.seed}.pt")
    if args.group == "G3":
        model = init_g34.init_g3(base, calib, CFG["r"], CFG["alpha"], CFG["rho"])
        # G3's initial A (a few MB): lets us verify that G3 and G4 share directions, and serves the H4 subspace comparison with G8/G9
        a_path = os.path.join(args.cache, f"g3A_{args.task}_{args.stage == 'lowres'}_s{args.seed}.pt")
        if not os.path.exists(a_path):
            os.makedirs(args.cache, exist_ok=True)
            torch.save({n.replace(init_g34.PREFIX, "").replace(".lora_A.default.weight", ""): p.detach().float().cpu()
                        for n, p in model.named_parameters() if "lora_A" in n}, f"{a_path}.{os.getpid()}.tmp")
            os.replace(f"{a_path}.{os.getpid()}.tmp", a_path)
    else:
        model, cache = init_g34.init_g4(base, calib, CFG["r"], CFG["alpha"], CFG["rho"],
                                        CFG["param_budget"], cache_path)
        extra["g3_rule_on_cache"] = alloc.eva_rank_count(cache["ratios"], CFG["r"] * len(targets),
                                                         round(CFG["r"] * CFG["rho"]))
    t3 = time.time()

    model.to(dev)
    ranks = alloc.lora_ranks(model)
    n_lora = alloc.lora_param_count(model)
    budget_dev = (n_lora - CFG["param_budget"]) / CFG["param_budget"]
    if args.group == "G4":
        assert abs(budget_dev) <= 0.01, f"G4 parameter count {n_lora} is {budget_dev:.2%} off the budget"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG["weight_decay"])
    n_ep = CFG["epochs"]["lowres" if args.stage == "lowres" else args.task]
    total = n_ep * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * total), total)

    model.train()
    step, grad_b0, losses, per_epoch = 0, None, [], []
    for _ in range(n_ep):
        for b in train_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(**b).loss
            loss.backward()
            if step == 0:   # gradient signal on B at initialisation (B = 0), before any update
                grad_b0 = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for n, p in model.named_parameters()
                                               if "lora_B" in n and p.grad is not None)))
            if step < CFG["loss_curve_steps"]:
                losses.append(loss.item())
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        per_epoch.append(evaluate(model, eval_dl, args.task, dev))
    t4 = time.time()

    res = dict(
        run_id=run_id, stage=args.stage, group=args.group, task=args.task, lr=args.lr, seed=args.seed,
        eval_split="dev" if args.stage in ("pilot", "tune") else "val", final=per_epoch[-1], per_epoch=per_epoch,
        step0_grad_B=grad_b0, loss_first_steps=losses, steps=step,
        lora_params=n_lora, head_params=alloc.head_param_count(model), budget_dev=budget_dev, ranks=ranks,
        time_s=dict(load=t1 - t0, head_warmup=t2 - t1, init=t3 - t2, train_eval=t4 - t3, total=t4 - t0),
        peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
        cfg=CFG, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        versions={m.__name__: m.__version__ for m in (torch, __import__("transformers"), __import__("peft"))},
        **extra,
    )
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"done {run_id}: {res['final']}  params={n_lora}  total={t4 - t0:.0f}s")


if __name__ == "__main__":
    main()
