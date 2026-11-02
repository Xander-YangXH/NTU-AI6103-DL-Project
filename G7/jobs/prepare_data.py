"""Prepare the shared GLUE disk datasets and RoBERTa-base model folder.

By default reuse the project's pinned parquet files and model. On another
machine use --download to fetch the same public model and data revisions.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download, snapshot_download


PROJECT = Path(__file__).resolve().parents[3]
MODEL_REPO = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DATA_REPO = "nyu-mll/glue"
DATA_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
EXPECTED = {"sst2": (67349, 872), "mrpc": (3668, 408), "rte": (2490, 277)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("GWEVA_DATA", PROJECT / "data/gweva-team")))
    parser.add_argument("--source-root", type=Path, default=PROJECT / "data/gweva/raw")
    parser.add_argument("--model-source", type=Path, default=PROJECT / "models/roberta-base")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        snapshot_download(
            MODEL_REPO, revision=MODEL_REVISION, local_dir=root / "roberta-base", max_workers=2,
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                            "tokenizer_config.json", "vocab.json", "merges.txt"],
        )
    else:
        source = args.model_source.expanduser().resolve(strict=True)
        target = root / "roberta-base"
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"{target} already points to a different model")
        elif target.exists():
            if target.resolve() != source:
                raise FileExistsError(f"{target} already exists; choose another --root")
        else:
            target.symlink_to(source, target_is_directory=True)

    counts = {}
    for task, expected in EXPECTED.items():
        splits = {}
        for split in ("train", "validation"):
            relative = f"{task}/{split}-00000-of-00001.parquet"
            if args.download:
                source = hf_hub_download(DATA_REPO, relative, repo_type="dataset",
                                          revision=DATA_REVISION, local_dir=root / "raw")
            else:
                source = args.source_root.expanduser() / relative
            splits[split] = Dataset.from_parquet(str(source), keep_in_memory=True,
                                                cache_dir=str(root / ".cache"))
        sizes = (len(splits["train"]), len(splits["validation"]))
        if sizes != expected:
            raise ValueError(f"Unexpected {task} train/validation sizes: {sizes}")
        DatasetDict(splits).save_to_disk(str(root / "glue" / task))
        counts[task] = dict(zip(("train", "validation"), sizes))
    manifest = {
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "dataset_repo": DATA_REPO, "dataset_revision": DATA_REVISION,
        "source_mode": "download" if args.download else "existing project files",
        "tasks": counts,
        "split": "G7/src/data.py; NumPy permutation seed 2026; no stratification",
    }
    (root / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"GWEVA_DATA": str(root), **manifest}, indent=2))


if __name__ == "__main__":
    main()
