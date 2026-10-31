import re

import torch
from peft import (
    EvaConfig,
    LoraConfig,
    get_peft_model,
    initialize_lora_eva_weights,
)

# target module regex: 72 linear layers, exclude classifier.dense
TARGET = r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)"
# base model prefix constant
PREFIX = "base_model.model."

# find lora location:
def target_names(model):
    # return list of matched names
    return [
        name
        for name, _ in model.named_modules()
        if re.fullmatch(TARGET, name)
    ]

# set lora config:
def build_lora_config(rank, alpha, rho):
    # build EvaConfig: rho, use_label_mask False
    eva_config = EvaConfig(
        rho=rho,
        use_label_mask=False
    )
    # build LoraConfig:
    lora_config = LoraConfig(
        # task_type SEQ_CLS(Sequence Classification)
        task_type="SEQ_CLS",
        # rank r, alpha, dropout 0.0
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        # target_modules regex
        target_modules=TARGET,
        # init_lora_weights "eva"
        init_lora_weights="eva",
        # eva_config
        eva_config=eva_config
    )
    # return LoraConfig
    return lora_config

# base + lora + eva:
def build_peft_model(base, calib, rank, alpha, rho):
    # get lora config
    lora_config = build_lora_config(rank, alpha, rho)

    # install lora
    peft_model = get_peft_model(
        base,
        lora_config,
        low_cpu_mem_usage=True
    )
    
    # launch eva
    initialize_lora_eva_weights(
        peft_model,
        calib,
        show_progress_bar=False
    )
    
    return peft_model

# extract lora a:
def extract_lora_a(peft_model):
    lora_a_dict = {}

    # iterate named modules
    for name, module in peft_model.named_modules():
        # check module has lora_A and "default" ∈ lora_A:
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            # detach, cpu, clone weight tensor
            weight = (
                module.lora_A["default"].weight
                .detach()
                .cpu()
                .clone()
            )
            # record name → tensor
            lora_a_dict[name] = weight

    return lora_a_dict

# check lora a row norms:
def lora_a_rank_stats(peft_model):
    # iterate named modules with lora_A:
    lora_a_dict = extract_lora_a(peft_model)
    all_norms = []

    for weight in lora_a_dict.values():
        # compute row L2 norm (dim 1)
        row_norms = torch.norm(weight, dim=1)
        # concat all layer row norms
        all_norms.append(row_norms)

    all_norms = torch.cat(all_norms)

    return {
        "mean": all_norms.mean().item(),
        "min": all_norms.min().item(),
        "max": all_norms.max().item(),
    }