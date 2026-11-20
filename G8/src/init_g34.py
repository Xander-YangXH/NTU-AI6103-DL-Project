"""G3 (EVA, PEFT built-in) and G4 (same EVA components, parameter-cost rank allocation) initialisation."""
import os
import re

import torch
from peft import EvaConfig, LoraConfig, get_eva_state_dict, get_peft_model, initialize_lora_eva_weights

from alloc import param_cost

# All 72 attention and FFN linear layers of RoBERTa-base; excludes classifier.dense.
TARGET = r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)"
PREFIX = "base_model.model."


def target_names(model):
    return [n for n, _ in model.named_modules() if re.fullmatch(TARGET, n)]


def lora_config(r, alpha, rho):
    return LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGET,
                      init_lora_weights="eva", eva_config=EvaConfig(rho=rho, use_label_mask=False))


def init_g3(base, calib, r, alpha, rho):
    """Original EVA: PEFT's incremental SVD and its rank-count redistribution (budget = r x 72 ranks)."""
    model = get_peft_model(base, lora_config(r, alpha, rho), low_cpu_mem_usage=True)
    initialize_lora_eva_weights(model, dataloader=calib, show_progress_bar=False)
    return model


@torch.no_grad()
def explained_variance_ratios(model, loader, comps):
    """Explained-variance ratio of each cached component: v^T C v / tr(C), C = centred covariance of the
    layer input over non-padding tokens of the calibration set (same definition as PEFT's IncrementalPCA)."""
    dev = next(model.parameters()).device
    mods = dict(model.named_modules())
    stats, handles, state = {}, [], {}

    def hook(name):
        def fn(_m, inp, _out):
            x = inp[0][state["mask"]].double()
            n, s, ss = stats.get(name, (0, 0.0, 0.0))
            stats[name] = (n + x.shape[0], s + x.sum(0), ss + x.T @ x)
        return fn

    for name in comps:
        handles.append(mods[name].register_forward_hook(hook(name)))
    model.eval()
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        state["mask"] = batch["attention_mask"].bool()
        model(**batch)
    for h in handles:
        h.remove()

    ratios = {}
    for name, v in comps.items():
        n, s, ss = stats[name]
        mu = s / n
        cov = (ss - n * torch.outer(mu, mu)) / (n - 1)
        vv = v.to(cov.device).double()
        ratios[name] = ((vv @ cov) * vv).sum(1) / torch.trace(cov)
    return {k: r.float().cpu() for k, r in ratios.items()}


def eva_cache(base, calib, cap, path):
    """Top-`cap` EVA components for every target layer (no redistribution) plus their explained-variance
    ratios. Cached per (task, seed) so G4 and the cross-checks reuse the same directions."""
    if os.path.exists(path):
        return torch.load(path)
    cfg = lora_config(cap, 1, 1.0)   # r = cap and rho = 1: every layer keeps all `cap` components
    comps = get_eva_state_dict(base, calib, peft_config=cfg, show_progress_bar=False)
    comps = {k: v.float().cpu() for k, v in comps.items()}
    ratios = explained_variance_ratios(base, calib, comps)
    cache = {"comps": comps, "ratios": ratios}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"   # two runs on one GPU may build the same cache at once
    torch.save(cache, tmp)
    os.replace(tmp, path)
    return cache


def init_g4(base, calib, r, alpha, rho, param_budget, cache_path):
    """Same EVA components as G3, ranks allocated by explained variance per parameter under a fixed
    parameter budget. Returns the PEFT model and the cache (for diagnostics)."""
    cap = round(r * rho)
    cache = eva_cache(base, calib, cap, cache_path)
    mods = dict(base.named_modules())
    costs = {n: mods[n].in_features + mods[n].out_features for n in cache["comps"]}
    ranks = param_cost(cache["ratios"], costs, param_budget, cap)
    # Every target layer must appear; a 0-row tensor tells PEFT to drop LoRA from that layer.
    state = {PREFIX + n: cache["comps"][n][:k].clone() for n, k in ranks.items()}
    model = get_peft_model(base, lora_config(r, alpha, rho), low_cpu_mem_usage=True)
    initialize_lora_eva_weights(model, eva_state_dict=state, show_progress_bar=False)
    return model, cache
