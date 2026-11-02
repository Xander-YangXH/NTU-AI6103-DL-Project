#!/usr/bin/env bash
#SBATCH --job-name=g56
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --output=g56-%j.log
set -euo pipefail
# Set partition/account/QoS with sbatch flags appropriate to your cluster.
# Submit one task at a time; the original 26-run queue took about10h total.
cd "${GWEVA_CODE:?Set GWEVA_CODE to the absolute G5G6 folder}"
bash jobs/run_queue.sh --tasks "${TASK:-rte}"
