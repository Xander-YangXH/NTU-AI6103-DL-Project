#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=g34
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# QoS "normal": 1 GPU, 10 CPUs, 30G RAM, 6 h per user. This job takes the GPU, 8 CPUs and 24G and runs NPAR
# training processes side by side on that one GPU; 2 CPUs and 6G stay free for a CPU-only job.
# Submit from ~/myproject/gweva-g34 (the queue is generated here, on the compute node, not on the head node):
#   sbatch --export=ALL,STAGE=tune,NPAR=2 jobs/g34.sbatch        # STAGE = pilot | tune | main | lowres
set -u
[ -z "${SLURM_JOB_ID:-}" ] && { echo "submit with sbatch, do not run on the login node"; exit 1; }
set +u   # conda's own activate.d scripts read unset variables (e.g. QT_XCB_GL_INTEGRATION)
module load anaconda
eval "$(conda shell.bash hook)"
conda activate gweva
set -u
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
# data sits next to the code folder: ~/myproject/gweva-g34 -> ~/myproject/gweva-data (override with GWEVA_DATA)
export GWEVA_DATA="${GWEVA_DATA:-$(dirname "$SLURM_SUBMIT_DIR")/gweva-data}"
echo "job $SLURM_JOB_ID on $(hostname)  stage=$STAGE  npar=${NPAR:-2}  start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python jobs/make_queue.py "$STAGE" || exit 1
bash jobs/run_queue.sh "jobs/queues/$STAGE.txt" "${NPAR:-2}"
echo "end=$(date)"
