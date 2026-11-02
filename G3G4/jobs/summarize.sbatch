#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=g34sum
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# CPU-only (no --gres): fits next to a running g34 job within the per-user limits.
# If the partition rejects jobs without a GPU, run jobs/summarize.py on your laptop after downloading results/ and cache/.
set -u
[ -z "${SLURM_JOB_ID:-}" ] && { echo "submit with sbatch, do not run on the login node"; exit 1; }
set +u   # conda's own activate.d scripts read unset variables (e.g. QT_XCB_GL_INTEGRATION)
module load anaconda
eval "$(conda shell.bash hook)"
conda activate gweva
set -u
cd "$SLURM_SUBMIT_DIR"
python jobs/summarize.py
