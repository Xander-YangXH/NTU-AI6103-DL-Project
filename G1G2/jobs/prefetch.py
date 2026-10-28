"""Download RoBERTa-base and GLUE SST-2/MRPC/RTE once and save them as plain local folders.

Run it on your own computer (the cluster's compute nodes have no internet), then upload the folder to
~/myproject/gweva-data on the cluster (next to the code folder). Windows PowerShell:
    $env:GWEVA_DATA = "D:\\Codex\\DL_assignment\\code\\gweva-data"; python jobs\\prefetch.py
"""
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

root = os.path.expanduser(os.environ.get("GWEVA_DATA", "~/gweva-data"))
expected = {"sst2": (67349, 872), "mrpc": (3668, 408), "rte": (2490, 277)}

# Copy the hub checkpoint file for file (no classification head inside), so that train.py creates a fresh head
# under each run's seed exactly as it would when loading from the hub. Saving a model that already had a head
# would freeze one random head for every seed.
snapshot_download("FacebookAI/roberta-base", local_dir=os.path.join(root, "roberta-base"),
                  allow_patterns=["config.json", "model.safetensors", "merges.txt", "vocab.json", "tokenizer.json",
                                  "tokenizer_config.json"])
for task, (n_train, n_val) in expected.items():
    ds = load_dataset("nyu-mll/glue", task)
    assert (len(ds["train"]), len(ds["validation"])) == (n_train, n_val), (task, ds)
    ds.save_to_disk(os.path.join(root, "glue", task))
    print(task, {k: len(v) for k, v in ds.items()})
print("saved to", root)
