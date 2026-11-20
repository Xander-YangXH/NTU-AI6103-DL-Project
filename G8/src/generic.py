"""G8 calibration data: the examples G3 calibrates on, with their text replaced by wikitext-103.

data.calib_loader picks a random subset of the training split for each seed. G8 picks the same subset and keeps
each example's exact token layout: total length, the single-sentence or sentence-pair format, and both segment
lengths. Only the tokens inside the segments are swapped for consecutive wikitext-103 tokens. The DataLoader
shuffles with the same generator, so batch i of G8 has the same shapes as batch i of G3. Content is then the only
difference between G3 and G8.
"""
import os

import numpy as np
from datasets import Dataset, load_from_disk

import data

# Written once by jobs/prefetch.py (wikitext-103-raw-v1, train split); override for a local test.
GENERIC_DIR = os.path.expanduser(os.environ.get("GWEVA_GENERIC", os.path.join(data.DATA_ROOT, "wikitext103")))
GENERIC_SEED = 2028   # team constants use 2026 (dev split) and 2027 (low-resource subset)


def segment_lengths(ids, cls, sep):
    """Segment lengths of a RoBERTa input: [cls] a [sep] -> [len(a)]; [cls] a [sep][sep] b [sep] -> [len(a), len(b)]."""
    assert ids[0] == cls and ids[-1] == sep, ids[:3]
    inner = ids[1:-1]
    if sep not in inner:
        return [len(inner)]
    k = inner.index(sep)
    assert inner[k + 1] == sep and sep not in inner[k + 2:], "unexpected sentence-pair layout"
    return [k, len(inner) - k - 2]


def build(segs, stream, cls, sep):
    """Fill a layout with tokens from the stream; returns the input ids and the number of tokens used."""
    parts, pos = [], 0
    for n in segs:
        parts.append(stream[pos:pos + n])
        pos += n
    ids = [cls] + parts[0] + [sep] if len(parts) == 1 else [cls] + parts[0] + [sep, sep] + parts[1] + [sep]
    return ids, pos


def token_stream(tokenizer, n_tokens, seed):
    """At least n_tokens consecutive wikitext-103 tokens from article lines in a seeded random order (headings and
    blank lines dropped)."""
    raw = load_from_disk(GENERIC_DIR)
    order = np.random.default_rng(seed).permutation(len(raw))
    out = []
    for start in range(0, len(order), 4096):
        texts = [t for t in list(raw.select(order[start:start + 4096])["text"])
                 if t.strip() and not t.lstrip().startswith("=")]
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            out.extend(ids)
        if len(out) >= n_tokens:
            return out
    raise RuntimeError(f"wikitext-103 has fewer than {n_tokens} tokens")


def generic_calib_loader(train_ds, tokenizer, size, batch_size, seed, num_workers=2):
    """Drop-in replacement for data.calib_loader: same subset, same layouts, same shuffle, wikitext content."""
    idx = np.random.default_rng(seed).permutation(len(train_ds))[: min(size, len(train_ds))]   # as data.calib_loader
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    layouts = [segment_lengths(list(ids), cls, sep) for ids in train_ds.select(idx)["input_ids"]]
    stream = token_stream(tokenizer, sum(map(sum, layouts)), GENERIC_SEED + seed)
    rows, pos = [], 0
    for segs in layouts:
        ids, used = build(segs, stream[pos:], cls, sep)
        rows.append(ids)
        pos += used
    ds = Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]})
    return data.make_loader(ds, tokenizer, batch_size, shuffle=True, seed=seed, num_workers=num_workers)
