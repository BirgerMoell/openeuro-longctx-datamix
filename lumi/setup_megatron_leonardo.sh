#!/bin/bash
# Run once from a Leonardo login node.
# Clones the Megatron fork and installs required Python packages into $WORK/megatron_venv.
# Compute nodes have no internet — all installs must happen here.

set -euo pipefail

MEGATRON_DIR=$WORK/NVIDIA-Megatron-LM
VENV=$WORK/megatron_venv

module purge
module load python/3.11.7
module load cuda/12.1

# ── Clone Megatron ────────────────────────────────────────────────────────────
if [ ! -d "$MEGATRON_DIR" ]; then
    echo "Cloning Megatron fork..."
    git clone https://github.com/luomajouni/NVIDIA-Megatron-LM.git "$MEGATRON_DIR"
    echo "Cloned to $MEGATRON_DIR"
else
    echo "Megatron already at $MEGATRON_DIR"
fi

# ── Create venv ───────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet --upgrade pip

# Core training deps
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers accelerate sentencepiece

# Megatron deps
pip install --quiet pybind11 regex six nltk
pip install --quiet tensorboard wandb

echo ""
echo "=== Checking installs ==="
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import transformers; print('transformers:', transformers.__version__)"

echo ""
echo "=== Setup complete ==="
echo "Megatron: $MEGATRON_DIR"
echo "Venv:     $VENV"
echo ""
echo "Next: transfer data files to \$WORK/data/ and submit the smoke test:"
echo "  sbatch lumi/slurm/train_smoke_test_leonardo.sbatch"
