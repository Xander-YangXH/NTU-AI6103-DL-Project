#!/bin/bash
# Run once on the login node (needs internet, no GPU): conda env with the team's pinned versions, then the
# model, GLUE and wikitext-103 saved to ../gweva-data. Reuse an existing env with CONDA_ENV=name.
#   bash jobs/setup.sh
set -eo pipefail   # no -u: conda's activate.d scripts read unset variables
cd "$(dirname "$0")/.."
ENV="${CONDA_ENV:-gweva}"
module load anaconda
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -y -n "$ENV" --override-channels -c conda-forge python=3.12 pip
fi
conda activate "$ENV"
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export GWEVA_DATA="${GWEVA_DATA:-$(dirname "$PWD")/gweva-data}"
python jobs/prefetch.py
python -c "import torch, transformers, peft, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__)"
echo "env $ENV ready, data in $GWEVA_DATA"
