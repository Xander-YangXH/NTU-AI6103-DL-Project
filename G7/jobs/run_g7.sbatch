#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=TC2N[03-08]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --job-name=G7-team
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
PROJECT=$(cd ../.. && pwd)
export GWEVA_DATA="${GWEVA_DATA:-$PROJECT/data/gweva-team}"
export GWEVA_PYTHON="${GWEVA_PYTHON:-$PROJECT/.venv/bin/python}"
export HF_HOME="$PROJECT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT/.cache"
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$GWEVA_PYTHON" -u jobs/run_experiments.py
