#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:10:00
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/flash-attn-install-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/flash-attn-install-%j.err

# Install flash-attn 2.8.3 from the prebuilt GitHub wheel (no compilation needed).
# The login node has internet; the wheel is for cu12+torch2.6+cxx11abiFALSE+cp311.
# Run this on the login node — no GPU needed for wheel installation:
#
#   source $WORK/megatron_venv/bin/activate
#   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps
#
# This sbatch is kept for cases where the install must go through the job scheduler.
# Submit: sbatch lumi/slurm/install_flash_attn_leonardo.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

module purge
module load python/3.11.7

source "$VENV/bin/activate"

echo "=== Installing flash-attn 2.8.3 (prebuilt wheel) — $(date) ==="
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo ""

# --no-deps: all runtime deps (torch, einops) are already in the venv
pip install "$WHEEL_URL" --no-deps 2>&1

echo ""
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
echo "=== Done — $(date) ==="
