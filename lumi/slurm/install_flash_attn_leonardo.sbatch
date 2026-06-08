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

# Build and install flash-attn 2.8.3 from source on a GPU compute node.
# Requires: source tarball pre-downloaded to $WORK/flash_attn-2.8.3.tar.gz
#
# Why source build: prebuilt wheels require GLIBC 2.32; Leonardo has GLIBC 2.28.
# Why gcc/12.2.0: system GCC 8.5 cannot compile PyTorch 2.6 C++ headers
#   (constexpr overflow in DispatchKeySet.h).
# Why cuda/12.6: torch was compiled with cu124; NVCC 12.6 is forward-compatible.
# Why TMPDIR=$WORK/tmp: pip moves the built wheel via os.rename(); without this,
#   pip uses /tmp (different filesystem from $WORK) and gets Errno 18.
#
# Submit: sbatch lumi/slurm/install_flash_attn_leonardo.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
TARBALL=$WORK/flash_attn-2.8.3.tar.gz
BUILD_DIR=$WORK/flash_attn_build_$SLURM_JOB_ID

module purge
module load gcc/12.2.0
module load cuda/12.6
module load python/3.11.7

# Set TMPDIR after module loads (python module resets it to /tmp).
# Must be on same filesystem as $WORK to avoid Errno 18 cross-device link.
mkdir -p "$WORK/tmp"
export TMPDIR="$WORK/tmp"

source "$VENV/bin/activate"

echo "=== Building flash-attn — $(date) ==="
echo "GCC:   $(gcc --version | head -1)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA:  $(nvcc --version | grep release)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "TMPDIR: $TMPDIR"
echo ""

mkdir -p "$BUILD_DIR"
tar xzf "$TARBALL" -C "$BUILD_DIR"
cd "$BUILD_DIR"/flash_attn-2.8.3

export MAX_JOBS=16
# --no-deps: compute nodes have no internet; torch/einops already in venv
# --no-cache-dir: prevent pip from using ~/.cache/pip (different filesystem)
pip install . --no-build-isolation --no-deps --no-cache-dir 2>&1

echo ""
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
echo "=== Done — $(date) ==="
