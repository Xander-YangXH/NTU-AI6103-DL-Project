#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=gweva-smoke
#SBATCH --output=smoke_%j.out
# GPU smoke test of every group (see tools/smoke_test.py). Submit from the repository root. Partition, QoS and node
# flags depend on the cluster; on NTU CCDS TC2 (L40S nodes):
#   sbatch -p MGPU-TC2 --qos=normal --nodelist=TC2N08 --export=ALL,VENV=$HOME/gweva/venv,GWEVA_DATA=$HOME/gweva/gweva-data tools/smoke_test.sbatch
# SMOKE_GROUPS picks the groups (default: all, including H1).
set -euo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "submit with sbatch"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers, peft, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__)"
python tools/smoke_test.py --groups ${SMOKE_GROUPS:-G1 G2 G3 G4 G8 G9 G7 G5G6 H1}
echo "end $(date)"
