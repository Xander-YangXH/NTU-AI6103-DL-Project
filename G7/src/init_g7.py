"""LoRA-GA initialisation, following the authors' ArB2r/stable implementation.

Source: Outsider565/LoRA-GA, c4cd5372c75b290924214b348008891f744512ef,
peft/src/peft/tuners/lora/layer.py::lora_ga_init.
"""
import re
from time import perf_counter

import torch
from peft import LoraConfig, get_peft_model


TARGET = r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)"
PREFIX = "base_model.model."


def target_names(model):
    return [name for name, _ in model.named_modules() if re.fullmatch(TARGET, name)]


def uniform_budget(model, names, rank):
    modules = dict(model.named_modules())
    return sum(rank * (modules[name].in_features + modules[name].out_features) for name in names)


def lora_ranks(model):
    return {name.replace(PREFIX, ""): module.lora_A["default"].weight.shape[0]
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") and "default" in module.lora_A}


def lora_param_count(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if "lora_A" in name or "lora_B" in name)


def head_param_count(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if p.requires_grad and "lora_" not in name)


def init_g7(base, calibration_loader, rank, alpha, gamma=16):
    """Estimate mean full-weight gradients, initialise A/B, and compensate W.

    Calibration runs in FP32/eval with the warmed-up classifier frozen.
    ``loss * batch_size`` removes minibatch averaging before the complete
    sum is divided by the actual number of calibration examples.
    """
    started = perf_counter()
    device = next(base.parameters()).device
    names = target_names(base)
    if device.type != "cuda" or len(names) != 72:
        raise ValueError("G7 requires a Slurm GPU and 72 RoBERTa target layers")
    targets = {name: base.get_submodule(name) for name in names}
    if any(module.weight.dtype != torch.float32 for module in targets.values()):
        raise ValueError("LoRA-GA calibration and compensation require FP32 weights")
    was_training = base.training
    original_requires_grad = [(p, p.requires_grad) for p in base.parameters()]
    for parameter, _ in original_requires_grad:
        parameter.requires_grad_(False)
    for module in targets.values():
        module.weight.requires_grad_(True)
    base.eval()
    base.zero_grad(set_to_none=True)
    sample_count, batch_count, loss_sum = 0, 0, 0.0
    first_batch, logits_before = None, None
    try:
        for batch in calibration_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            size = batch["labels"].shape[0]
            output = base(**batch)
            (output.loss * size).backward()
            if first_batch is None:
                first_batch = {key: value.detach() for key, value in batch.items()}
                logits_before = output.logits.detach().clone()
            sample_count += size
            batch_count += 1
            loss_sum += output.loss.detach().item() * size
        if not sample_count:
            raise ValueError("Empty calibration set")
        for name, module in targets.items():
            if module.weight.grad is None:
                raise ValueError(f"No calibration gradient for {name}")
            module.weight.grad.div_(sample_count)
            if not torch.isfinite(module.weight.grad).all():
                raise ValueError(f"Nonfinite calibration gradient for {name}")
    finally:
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)

    gradients = {name: module.weight.grad.detach() for name, module in targets.items()}
    base.zero_grad(set_to_none=True)
    config = LoraConfig(task_type="SEQ_CLS", r=rank, lora_alpha=alpha, lora_dropout=0.0,
                        target_modules=TARGET, init_lora_weights=True, bias="none")
    model = get_peft_model(base, config)
    model.eval()
    layer_details = []
    with torch.no_grad():
        for name, gradient in gradients.items():
            layer = model.get_base_model().get_submodule(name)
            u, singular_values, v = torch.svd_lowrank(gradient, q=4 * rank, niter=4)
            scale = gradient.shape[0] ** 0.25 / gamma ** 0.5
            a = (v[:, :rank].T * scale).contiguous()
            b = (u[:, rank:2 * rank] * scale).contiguous()
            layer.lora_A["default"].weight.copy_(a)
            layer.lora_B["default"].weight.copy_(b)
            eta = layer.scaling["default"]
            original = layer.get_base_layer().weight.detach().clone()
            offset = eta * (b @ a)
            layer.get_base_layer().weight.sub_(offset)
            error = (layer.get_base_layer().weight + offset - original).abs().max().item()
            if not torch.isfinite(layer.get_base_layer().weight).all() or error > 1e-6:
                raise ValueError(f"Invalid weight compensation for {name}: {error}")
            layer_details.append(dict(
                name=name, shape=list(gradient.shape), rank=rank, scaling=eta,
                gradient_frobenius=gradient.norm().item(),
                singular_values_top_2r=singular_values[:2 * rank].tolist(),
                a_frobenius=a.norm().item(), b_frobenius=b.norm().item(),
                offset_frobenius=offset.norm().item(), weight_compensation_max_abs_error=error,
            ))
        logits_after = model(**first_batch).logits
        logits_error = (logits_after - logits_before).abs().max().item()
        if not torch.isfinite(logits_after).all() or logits_error > 1e-3:
            raise ValueError(f"Initial compensation changed logits by {logits_error}")
    model.train(was_training)
    details = dict(
        implementation="LoRA-GA authors c4cd537; pure singular vectors; stable scaling on A and B",
        direction="ArB2r", scale="stable", stable_gamma=gamma,
        calibration_mode="FP32; eval; classifier frozen; labels retained",
        calibration_samples=sample_count, calibration_batches=batch_count,
        calibration_mean_loss=loss_sum / sample_count,
        initial_logits_max_abs_error=logits_error,
        initialization_seconds=perf_counter() - started, layers=layer_details,
    )
    return model, details
