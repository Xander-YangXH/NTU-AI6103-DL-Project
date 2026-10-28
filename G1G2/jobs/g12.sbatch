#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=TC2N0[1-6]
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=g12
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# Same as G3G4/jobs/g34.sbatch, except: only TC2N07/N08 (L40S), and the venv ../venv instead of conda env gweva.
# Submit from ~/gweva/G1G2:
#   sbatch --export=ALL,STAGE=tune,NPAR=2 jobs/g12.sbatch        # STAGE = pilot | tune | main | lowres
set -u
[ -z "${SLURM_JOB_ID:-}" ] && { echo "submit with sbatch, do not run on the login node"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
# venv on top of conda env ecomq (torch 2.11.0+cu128) with transformers 5.21.0 / peft 0.21.2 / datasets 5.1.0
source "$(dirname "$SLURM_SUBMIT_DIR")/venv/bin/activate"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# data sits next to the code folder: ~/gweva/G1G2 -> ~/gweva/gweva-data (override with GWEVA_DATA)
export GWEVA_DATA="${GWEVA_DATA:-$(dirname "$SLURM_SUBMIT_DIR")/gweva-data}"
echo "job $SLURM_JOB_ID on $(hostname)  stage=$STAGE  npar=${NPAR:-2}  start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "not an L40S node: $(hostname)"; exit 2; }
python jobs/make_queue.py "$STAGE" || exit 1
bash jobs/run_queue.sh "jobs/queues/$STAGE.txt" "${NPAR:-2}"
[ "$STAGE" = main ] || [ "$STAGE" = lowres ] && python jobs/summarize.py
echo "end=$(date)"
