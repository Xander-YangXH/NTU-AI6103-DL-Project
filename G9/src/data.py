"""GLUE data for G3/G4: fixed dev split, low-resource subset, tokenisation and loaders.

The split constants below must be identical for all six members (agree with A once, then never change).
"""
import os

import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Local copies written once by jobs/prefetch.py, so jobs never touch the network.
DATA_ROOT = os.path.expanduser(os.environ.get("GWEVA_DATA", "~/gweva-data"))
TASK_KEYS = {"sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"), "rte": ("sentence1", "sentence2")}
SPLIT_SEED = 2026   # team-wide constant
DEV_FRAC = 0.10     # 10% of the official train set is held out for model selection
LOWRES_N = 1000     # low-resource SST-2 subset, drawn from the remaining 90%


def load_glue(task):
    return load_from_disk(os.path.join(DATA_ROOT, "glue", task))


def load_splits(task, tokenizer, max_len):
    """Return {"train": 90% of train, "dev": 10% of train, "val": official validation, "lowres": 1k of train}."""
    raw = load_glue(task)
    k1, k2 = TASK_KEYS[task]

    def tok(batch):
        return tokenizer(batch[k1], batch[k2] if k2 else None, truncation=True, max_length=max_len)

    drop = [c for c in raw["train"].column_names if c != "label"]
    # keep_in_memory: no cache files, so two processes on one GPU never write the same file
    enc = raw.map(tok, batched=True, remove_columns=drop, keep_in_memory=True).rename_column("label", "labels")

    n = len(enc["train"])
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_dev = int(round(DEV_FRAC * n))
    dev_idx, train_idx = perm[:n_dev], perm[n_dev:]
    lowres_idx = np.random.default_rng(SPLIT_SEED + 1).permutation(train_idx)[:LOWRES_N]
    return {
        "train": enc["train"].select(train_idx),
        "dev": enc["train"].select(dev_idx),
        "val": enc["validation"],
        "lowres": enc["train"].select(lowres_idx),
        "indices": {"dev": dev_idx.tolist(), "lowres": lowres_idx.tolist()},
    }


class DictCollator:
    """Dynamic padding that returns a plain dict: PEFT's EVA hooks reject transformers' BatchEncoding."""

    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        return dict(self.pad(features))


def make_loader(ds, tokenizer, batch_size, shuffle, seed, num_workers=2, drop_labels=False):
    if drop_labels:
        ds = ds.remove_columns("labels")
    gen = None
    if shuffle:
        import torch
        gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=num_workers,
        collate_fn=DictCollator(tokenizer),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def calib_loader(train_ds, tokenizer, size, batch_size, seed, num_workers=2):
    """Calibration data for EVA: a fixed random subset of the training split, without labels.

    Labels are removed because sequence-classification labels have shape (B,), while PEFT's default
    label mask expects token-level labels of shape (B, T).
    """
    idx = np.random.default_rng(seed).permutation(len(train_ds))[: min(size, len(train_ds))]
    return make_loader(train_ds.select(idx), tokenizer, batch_size, shuffle=True, seed=seed,
                       num_workers=num_workers, drop_labels=True)
