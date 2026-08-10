#!/bin/bash
#SBATCH --job-name=dsa-rt-8k
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.out
#SBATCH --error=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.err
set -euo pipefail
export DSADIR=${DSADIR:-$SLURM_SUBMIT_DIR}
export SEQ_LENGTH=8192 CP_SIZE=1 ROTARY_BASE=64000000 RUN_GPU_TESTS=1
export OUT=${DSA_OUT:-/scratch/project_465002530/users/bmoell/longctx-extend/output_dsa_sparse8k_roundtrip}
export DATA_BLEND_FILE=${DATA_BLEND_FILE:-/scratch/project_465002530/users/bmoell/longctx-extend/jouni_blend.txt}
exec bash "$DSADIR/lumi/dsa_sparse_roundtrip.sh"
