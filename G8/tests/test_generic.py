"""Check that G8's calibration set keeps G3's exact token layouts (CPU, offline, a few seconds).

    GWEVA_DATA=... python tests/test_generic.py
Needs the prefetched data folder (roberta-base, glue/<task>, wikitext103). For every task: same examples as
data.calib_loader, identical lengths, identical special-token positions, and no task tokens left in the segments.
"""
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data  # noqa: E402
import generic  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.path.join(data.DATA_ROOT, "roberta-base"))
cls, sep = tok.cls_token_id, tok.sep_token_id
for task in ["sst2", "mrpc", "rte"]:
    train = data.load_splits(task, tok, 128)["train"]
    seed, size = 3, 64
    g3_batches = list(data.calib_loader(train, tok, size, 16, seed, num_workers=0))
    g8_batches = list(generic.generic_calib_loader(train, tok, size, 16, seed, num_workers=0))
    assert len(g3_batches) == len(g8_batches)
    same_tokens = total = 0
    for b3, b8 in zip(g3_batches, g8_batches):
        assert set(b8) == {"input_ids", "attention_mask"}, set(b8)
        assert b3["input_ids"].shape == b8["input_ids"].shape
        assert (b3["attention_mask"] == b8["attention_mask"]).all()
        special3 = (b3["input_ids"] == cls) | (b3["input_ids"] == sep)
        special8 = (b8["input_ids"] == cls) | (b8["input_ids"] == sep)
        assert (special3 == special8).all(), "special-token positions differ"
        body = b3["attention_mask"].bool() & ~special3
        same_tokens += int((b3["input_ids"][body] == b8["input_ids"][body]).sum())
        total += int(body.sum())
    print(f"{task}: {len(g8_batches)} batches, layouts identical, tokens equal to G3 at {same_tokens}/{total} positions")
    assert same_tokens / total < 0.05
print("ok")
