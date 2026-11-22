"""One team-protocol G7 run: head warm-up, LoRA-GA, training, last-epoch JSON.

The data and training loop follow shared source commit 26616379899f0abce7d693e227ad4e03138312ab.
Needs a CUDA GPU (Slurm or not). Completed result JSONs are skipped on resubmission.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import data
import init_g7


CFG = dict(
    model=os.path.join(data.DATA_ROOT, "roberta-base"), n_targets=72, r=8, alpha=8, rho=2.0,
    param_budget=1_327_104, max_len=128, batch_size=32, warmup_ratio=0.06, weight_decay=0.0,
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
def evaluate(model, loader, task, dev, return_predictions=False):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {key: value.to(dev) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    model.train()
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    result = metrics(task, preds, labels)
    if return_predictions:
        return result, {"predictions": preds.tolist(), "labels": labels.tolist()}
    return result


def warm_up_head(model, loader, epochs, lr, dev):
    """Reference warm-up: frozen encoder, model.train(), AdamW's default wd=0.01."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(dev) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_torch(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["tune", "main"])
    parser.add_argument("--task", required=True, choices=list(data.TASK_KEYS))
    parser.add_argument("--group", choices=["G7"], default="G7")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--cache", default="cache")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("G7 training and inference must run on a CUDA GPU")
    torch.set_num_threads(2)
    run_id = f"{args.stage}_{args.group}_{args.task}_lr{args.lr:g}_s{args.seed}"
    out_path = Path(args.out) / args.stage / (run_id + ".json")
    if out_path.exists():
        print(f"skip {run_id}: done", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) / run_id
    cache_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(CFG["model"], local_files_only=True)
    splits = data.load_splits(args.task, tokenizer, CFG["max_len"])
    train_ds = splits["train"]
    workers = CFG["num_workers"]
    # One generator is consumed continuously by warm-up and all training epochs.
    train_dl = data.make_loader(train_ds, tokenizer, CFG["batch_size"], True, args.seed, workers)
    dev_dl = data.make_loader(splits["dev"], tokenizer, 128, False, args.seed, workers)
    calibration_indices = np.random.default_rng(args.seed).permutation(len(train_ds))[:CFG["calib_size"]]
    calibration = data.make_loader(train_ds.select(calibration_indices), tokenizer,
                                   CFG["calib_batch_size"], True, args.seed, workers)
    base = AutoModelForSequenceClassification.from_pretrained(
        CFG["model"], num_labels=2, local_files_only=True, torch_dtype=torch.float32,
    ).to(dev)
    targets = init_g7.target_names(base)
    assert len(targets) == CFG["n_targets"], len(targets)
    assert init_g7.uniform_budget(base, targets, CFG["r"]) == CFG["param_budget"]
    t1 = time.time()
    warm_up_head(base, train_dl, CFG["head_warmup_epochs"], CFG["head_lr"], dev)
    t2 = time.time()
    model, init_details = init_g7.init_g7(base, calibration, CFG["r"], CFG["alpha"])
    model.peft_config["default"].save_pretrained(cache_path)
    save_torch(snapshot(model), cache_path / "initial_trainable.pt")
    t3 = time.time()

    ranks = init_g7.lora_ranks(model)
    n_lora, n_head = init_g7.lora_param_count(model), init_g7.head_param_count(model)
    assert n_lora == CFG["param_budget"] and n_head == 592_130, (n_lora, n_head)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG["weight_decay"])
    n_epochs = CFG["epochs"][args.task]
    total = n_epochs * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(CFG["warmup_ratio"] * total), total)
    model.train()
    step, grad_b0, losses, per_epoch = 0, None, [], []
    for epoch in range(n_epochs):
        loss_sum = 0.0
        for batch in train_dl:
            batch = {key: value.to(dev) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            if step == 0:
                grad_b0 = float(torch.sqrt(sum(
                    p.grad.float().pow(2).sum() for name, p in model.named_parameters()
                    if "lora_B" in name and p.grad is not None)))
            if step < CFG["loss_curve_steps"]:
                losses.append(loss.item())
            loss_sum += loss.detach().item() * batch["labels"].numel()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        # The reference evaluates once per epoch. Use dev for these diagnostics
        # and reserve official validation for the last-epoch main model.
        per_epoch.append(evaluate(model, dev_dl, args.task, dev))
        print(json.dumps(dict(event="epoch", run_id=run_id, epoch=epoch + 1,
                              train_loss=loss_sum / len(train_ds), dev=per_epoch[-1])), flush=True)
    t4 = time.time()

    predictions = None
    final = per_epoch[-1]
    if args.stage == "main":
        val_dl = data.make_loader(splits["val"], tokenizer, 128, False, args.seed, workers)
        final, predictions = evaluate(model, val_dl, args.task, dev, return_predictions=True)
    t5 = time.time()
    save_torch(snapshot(model), cache_path / "last_trainable.pt")

    result = dict(
        run_id=run_id, stage=args.stage, group=args.group, task=args.task, lr=args.lr, seed=args.seed,
        eval_split="dev" if args.stage == "tune" else "val", final=final, per_epoch=per_epoch,
        per_epoch_eval_split="dev", reported_epoch=n_epochs,
        step0_grad_B=grad_b0, loss_first_steps=losses, steps=step,
        lora_params=n_lora, head_params=n_head,
        budget_dev=(n_lora - CFG["param_budget"]) / CFG["param_budget"], ranks=ranks,
        time_s=dict(load=t1-t0, head_warmup=t2-t1, init=t3-t2, train_eval=t4-t3,
                    final_eval=t5-t4, total=t5-t0),
        peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
        cfg=CFG, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name(),
        versions={module.__name__: module.__version__ for module in
                  (torch, __import__("transformers"), __import__("peft"))},
        init_details=init_details,
        data_split=dict(seed=data.SPLIT_SEED, dev_fraction=data.DEV_FRAC,
                        train_n=len(train_ds), dev_n=len(splits["dev"]), val_n=len(splits["val"])),
    )
    if predictions is not None:
        result["predictions"] = predictions
    temporary = out_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=1) + "\n")
    temporary.replace(out_path)
    print(f"done {run_id}: {final} params={n_lora} total={t5-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
