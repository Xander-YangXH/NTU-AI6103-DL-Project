"""G1 (plain LoRA) and G2 (magnitude control: random orthonormal A with EVA's row norm) initialisation."""
import re

import torch
from peft import LoraConfig, get_peft_model

# All 72 attention and FFN linear layers of RoBERTa-base; excludes classifier.dense (same as G3G4/src/init_g34.py).
TARGET = r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)"
PREFIX = "base_model.model."
# PEFT's EVA (whiten=False) copies the top PCA components into lora_A, so every row of EVA's A has norm 1.
EVA_ROW_NORM = 1.0


def target_names(model):
    return [n for n, _ in model.named_modules() if re.fullmatch(TARGET, n)]


def lora_config(r, alpha):
    # init_lora_weights=True: A ~ kaiming_uniform(a=sqrt(5)), B = 0 (PEFT default)
    return LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGET,
                      init_lora_weights=True)


def init_g1(base, r, alpha):
    """Plain LoRA: Kaiming-uniform A, B = 0, uniform rank r."""
    return get_peft_model(base, lora_config(r, alpha))


@torch.no_grad()
def init_g2(base, r, alpha, seed, row_norm=EVA_ROW_NORM):
    """Magnitude control: like G1, but each A has r random orthonormal rows scaled to EVA's row norm.

    The directions come from a private generator, so the global RNG (dropout, etc.) is consumed exactly as in G1
    and G1/G2 runs with the same seed differ only in A.
    """
    model = get_peft_model(base, lora_config(r, alpha))
    gen = torch.Generator().manual_seed(seed)
    for name, m in sorted(model.named_modules(), key=lambda t: t[0]):
        if hasattr(m, "lora_A") and "default" in m.lora_A:
            w = m.lora_A["default"].weight   # (r, d_in)
            q, _ = torch.linalg.qr(torch.randn(w.shape[1], w.shape[0], generator=gen, dtype=torch.float64))
            w.copy_((q.T * row_norm).to(w.dtype))
    return model


@torch.no_grad()
def a_row_norms(peft_model):
    """Row norms of every lora_A: G1 ~ 0.577 (Kaiming-uniform, 1/sqrt(3)), G2 = EVA = 1."""
    norms = torch.cat([m.lora_A["default"].weight.float().norm(dim=1).cpu()
                       for _, m in peft_model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A])
    return {"mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max())}
