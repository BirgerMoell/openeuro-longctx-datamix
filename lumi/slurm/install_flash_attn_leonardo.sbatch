#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/flash-attn-install-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/flash-attn-install-%j.err

# Build and install flash-attn 2.8.3 from source on a compute node.
# Compute nodes have no internet but have GPU + CUDA, so compilation works.
# Source tarball must be pre-downloaded to $WORK (done on login node).
#
# Submit: sbatch lumi/slurm/install_flash_attn_leonardo.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
TARBALL=$WORK/flash_attn-2.8.3.tar.gz
BUILD_DIR=/tmp/flash_attn_build_$SLURM_JOB_ID

module purge
module load python/3.11.7
module load cuda/12.1

source "$VENV/bin/activate"

echo "=== Building flash-attn — $(date) ==="
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA: $(nvcc --version | grep release)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo ""

mkdir -p "$BUILD_DIR"
tar xzf "$TARBALL" -C "$BUILD_DIR"
cd "$BUILD_DIR"/flash_attn-2.8.3

export MAX_JOBS=16
# --no-deps: compute nodes have no internet; all deps (einops, torch) are already in the venv
pip install . --no-build-isolation --no-deps 2>&1

echo ""
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
echo "=== Done — $(date) ==="
