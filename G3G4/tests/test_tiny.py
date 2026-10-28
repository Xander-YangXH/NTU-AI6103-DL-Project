"""Offline end-to-end check of G3/G4 on a tiny RoBERTa with synthetic GLUE-shaped data (CPU, about 1 minute).

Run from the repo root:  python tests/test_tiny.py
It patches the hub calls in train.py, so it needs no network and no GPU.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import alloc  # noqa: E402
import data  # noqa: E402
import init_g34  # noqa: E402
import train  # noqa: E402

VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{f"w{i}": i + 4 for i in range(60)}}


def tiny_tokenizer():
    t = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")


def fake_glue(task):
    rng = np.random.default_rng(0)

    def sent(y):
        # label-dependent token distribution, so the task is learnable
        return " ".join(f"w{int(rng.integers(0, 30) + 30 * y)}" for _ in range(rng.integers(5, 20)))

    def split(n):
        labels = rng.integers(0, 2, n)
        cols = {"sentence": [sent(y) for y in labels]} if task in (None, "sst2") else \
            {"sentence1": [sent(y) for y in labels], "sentence2": [sent(y) for y in labels]}
        return Dataset.from_dict({**cols, "label": labels.tolist(), "idx": list(range(n))})

    return DatasetDict(train=split(1200), validation=split(200))


def tiny_model(*_a, **_k):
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=len(VOCAB), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=256, max_position_embeddings=80, num_labels=2, pad_token_id=0)
    return RobertaForSequenceClassification(cfg)


def main():
    train.AutoTokenizer.from_pretrained = lambda *_a, **_k: tiny_tokenizer()
    train.AutoModelForSequenceClassification.from_pretrained = tiny_model
    data.load_glue = fake_glue
    # tiny model: 2 layers x 6 targets; uniform r = 8 costs 8 * (4*128 + 2*320) * 2
    train.CFG.update(n_targets=12, param_budget=18_432, epochs={"sst2": 2, "lowres": 2}, calib_size=256,
                     head_warmup_epochs=1, num_workers=0)

    tmp = tempfile.mkdtemp()
    out, cache = os.path.join(tmp, "results"), os.path.join(tmp, "cache")
    for group in ("G3", "G4"):
        sys.argv = ["train.py", "--stage", "tune", "--task", "sst2", "--group", group, "--lr", "1e-3",
                    "--seed", "0", "--out", out, "--cache", cache]
        train.main()
    sys.argv[-5:] = sys.argv[-5:]  # rerun G4: must be skipped
    train.main()

    g3 = json.load(open(os.path.join(out, "tune", "tune_G3_sst2_lr0.001_s0.json")))
    g4 = json.load(open(os.path.join(out, "tune", "tune_G4_sst2_lr0.001_s0.json")))
    c = torch.load(os.path.join(cache, "eva16_sst2_False_s0.pt"))

    # G3: PEFT redistributes a rank budget of r x 12, at most r*rho per layer
    assert sum(g3["ranks"].values()) == 8 * 12 and max(g3["ranks"].values()) <= 16, g3["ranks"]
    # G4: parameter budget met within the cheapest layer cost, ranks are capped
    assert abs(g4["lora_params"] - 18_432) < 128, g4["lora_params"]
    assert max(g4["ranks"].values()) <= 16
    # cached ratios are valid fractions of total variance and roughly decreasing; neighbours may swap
    # slightly because PEFT's incremental low-rank PCA stops early (EVA itself only uses counts)
    for r in c["ratios"].values():
        assert torch.all(r > 0) and float(r.sum()) <= 1 + 1e-4 and r[0] > r[-1], r
    # EVA's rule re-implemented on the cache should give nearly the same allocation as PEFT's G3
    l1 = sum(abs(g3["ranks"].get(k, 0) - v) for k, v in g4["g3_rule_on_cache"].items())
    for g in (g3, g4):
        assert g["step0_grad_B"] > 0 and len(g["loss_first_steps"]) > 0
    print("G3 ranks", g3["ranks"])
    print("G4 ranks", g4["ranks"])
    print("G3 params", g3["lora_params"], f"({g3['budget_dev']:+.1%})", "| G4 params", g4["lora_params"],
          f"({g4['budget_dev']:+.2%})")
    print("L1 distance between PEFT's G3 allocation and EVA's rule on the cache:", l1)
    print("dev acc G3", g3["final"]["acc"], "G4", g4["final"]["acc"])

    # G4 adapters must hold exactly the cached components, B must start at zero
    base = tiny_model()
    model, _ = init_g34.init_g4(base, data.calib_loader(
        data.load_splits("sst2", tiny_tokenizer(), 128)["train"], tiny_tokenizer(), 256, 16, 0, 0),
        8, 8, 2.0, 18_432, os.path.join(cache, "eva16_sst2_False_s0.pt"))
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and "default" in m.lora_A:
            key = name.replace(init_g34.PREFIX, "")
            k = m.lora_A["default"].weight.shape[0]
            assert torch.allclose(m.lora_A["default"].weight, c["comps"][key][:k])
            assert torch.count_nonzero(m.lora_B["default"].weight) == 0
    print("all checks passed")


if __name__ == "__main__":
    main()
