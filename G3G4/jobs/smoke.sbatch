#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=g34smoke
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# One-off check on a compute node (compute nodes have no internet): GPU and driver, CUDA from PyTorch,
# the offline unit test, and the model/data uploaded next to the code folder (~/myproject/gweva-data).
set -u
[ -z "${SLURM_JOB_ID:-}" ] && { echo "submit with sbatch, do not run on the login node"; exit 1; }
set +u   # conda's own activate.d scripts read unset variables (e.g. QT_XCB_GL_INTEGRATION)
module load anaconda
eval "$(conda shell.bash hook)"
conda activate gweva
set -u
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
# data sits next to the code folder: ~/myproject/gweva-g34 -> ~/myproject/gweva-data (override with GWEVA_DATA)
export GWEVA_DATA="${GWEVA_DATA:-$(dirname "$SLURM_SUBMIT_DIR")/gweva-data}"
echo "node=$(hostname)  start=$(date)  GWEVA_DATA=$GWEVA_DATA"
nvidia-smi
python -c "import torch, transformers, peft, datasets; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__); x = torch.ones(1, device='cuda'); print('CUDA result', (x + 1).item())"
python tests/test_tiny.py 2>&1 | tail -6
python - <<'EOF'
import os
from datasets import load_from_disk
root = os.path.expanduser(os.environ.get("GWEVA_DATA", "~/gweva-data"))
assert os.path.exists(os.path.join(root, "roberta-base", "model.safetensors")), f"model missing under {root}"
for task, (n_train, n_val) in {"sst2": (67349, 872), "mrpc": (3668, 408), "rte": (2490, 277)}.items():
    d = load_from_disk(os.path.join(root, "glue", task))
    assert (len(d["train"]), len(d["validation"])) == (n_train, n_val), (task, d)
    print(task, "ok", n_train, n_val)
print("data ok:", root)
EOF
echo "end=$(date)"
