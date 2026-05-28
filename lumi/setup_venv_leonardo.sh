#!/bin/bash
# Run once from a Leonardo login node to create the venv in $WORK.
# Compute nodes have no internet access, so installs must happen here.

set -euo pipefail

VENV=$WORK/ruler_venv

module purge
module load python/3.11.7

if [ -d "$VENV" ]; then
    echo "Venv already exists at $VENV — delete it first if you want a fresh install."
    exit 0
fi

echo "Creating venv at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet transformers accelerate "lm_eval[ruler]>=0.4.4" nltk rouge_score

echo ""
echo "Done. Venv ready at $VENV"
echo "Test it with: sbatch lumi/slurm/test_leonardo.sbatch"
