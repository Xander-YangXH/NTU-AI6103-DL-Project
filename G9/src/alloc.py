"""Rank allocation rules and parameter counting for G3 (EVA, per rank) and G4 (per parameter)."""
import torch


def eva_rank_count(ratios, rank_budget, cap):
    """EVA's rule (as in PEFT): take the globally top-`rank_budget` components by explained-variance ratio,
    at most `cap` per layer. Used only to cross-check PEFT's built-in G3 allocation."""
    items = [(float(v), name) for name, r in ratios.items() for v in r[:cap]]
    items.sort(key=lambda t: -t[0])
    ranks = {name: 0 for name in ratios}
    for _, name in items[:rank_budget]:
        ranks[name] += 1
    return ranks


def param_cost(ratios, costs, param_budget, cap):
    """G4's rule: rank components by explained-variance ratio per parameter (ratio / (d_in + d_out)) and
    take them greedily while the parameter budget allows. A component that no longer fits is skipped and
    cheaper ones are still considered, so the leftover budget is smaller than the cheapest layer cost.
    Within a layer all components cost the same, so each layer always receives a prefix of its components."""
    items = [(float(v) / costs[name], i, name) for name, r in ratios.items() for i, v in enumerate(r[:cap])]
    items.sort(key=lambda t: (-t[0], t[1]))
    ranks = {name: 0 for name in ratios}
    left = param_budget
    for _, _, name in items:
        if costs[name] <= left:
            ranks[name] += 1
            left -= costs[name]
    return ranks


def lora_ranks(peft_model, adapter="default"):
    """Rank actually used by each LoRA layer after initialisation (layers with rank 0 are absent)."""
    out = {}
    for name, m in peft_model.named_modules():
        if hasattr(m, "lora_A") and adapter in m.lora_A:
            out[name.replace("base_model.model.", "")] = m.lora_A[adapter].weight.shape[0]
    return out


def lora_param_count(peft_model):
    """Trainable LoRA parameters only (the classification head is counted separately)."""
    return sum(p.numel() for n, p in peft_model.named_parameters() if "lora_A" in n or "lora_B" in n)


def head_param_count(peft_model):
    return sum(p.numel() for n, p in peft_model.named_parameters()
               if p.requires_grad and "lora_" not in n)


def uniform_budget(model, target_names, r):
    """Parameter budget of uniform rank r on the given layers: sum of r * (d_in + d_out)."""
    mods = dict(model.named_modules())
    return sum(r * (mods[n].in_features + mods[n].out_features) for n in target_names)


@torch.no_grad()
def subspace_overlap(a, b):
    """Mean squared cosine of principal angles between row spaces of a (k1 x d) and b (k2 x d), in [0, 1]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    qa = torch.linalg.qr(a.double().T)[0]
    qb = torch.linalg.qr(b.double().T)[0]
    return float((qa.T @ qb).pow(2).sum() / min(qa.shape[1], qb.shape[1]))
