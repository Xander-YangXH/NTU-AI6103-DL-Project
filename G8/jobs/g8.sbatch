#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=g8
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# Same layout as G3G4/jobs/g34.sbatch: one GPU, NPAR training processes side by side, finished runs skipped.
# Submit from the G8 folder (data in ../gweva-data unless GWEVA_DATA is set; conda env from CONDA_ENV, default gweva):
#   sbatch --export=ALL,STAGE=pilot,NPAR=1 jobs/g8.sbatch     # one RTE run, about 3 minutes
#   sbatch --export=ALL,STAGE=main,NPAR=2 jobs/g8.sbatch      # the 13 runs
#   sbatch --export=ALL,STAGE=all,NPAR=2 jobs/g8.sbatch       # main, then lowres (5 short runs), then a results zip
set -u
[ -z "${SLURM_JOB_ID:-}" ] && { echo "submit with sbatch, do not run on the login node"; exit 1; }
set +u   # conda's own activate.d scripts read unset variables (e.g. QT_XCB_GL_INTEGRATION)
module load anaconda || exit 1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-gweva}" || exit 1
set -u
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
export GWEVA_DATA="${GWEVA_DATA:-$(dirname "$SLURM_SUBMIT_DIR")/gweva-data}"
echo "job $SLURM_JOB_ID on $(hostname)  stage=$STAGE  npar=${NPAR:-2}  start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
STAGES="$STAGE"
[ "$STAGE" = all ] && STAGES="main lowres"
for st in $STAGES; do
  python jobs/make_queue.py "$st" || exit 1
  bash jobs/run_queue.sh "jobs/queues/$st.txt" "${NPAR:-2}"
done
# one archive to send back: result JSONs, G8's initial A matrices and the logs (no model weights are saved)
python jobs/pack_results.py "g8_${STAGE}_${SLURM_JOB_ID}.zip"
echo "end=$(date)"
