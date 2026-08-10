#!/bin/bash
#SBATCH --job-name=dsa-rt-64k
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.out
#SBATCH --error=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.err
set -euo pipefail
export DSADIR=${DSADIR:-$SLURM_SUBMIT_DIR}
export SEQ_LENGTH=65536 CP_SIZE=2 ROTARY_BASE=128000000 RUN_GPU_TESTS=0
export OUT=${DSA_OUT:-/scratch/project_465002530/users/bmoell/longctx-extend/output_dsa_sparse64k_cp2_roundtrip}
export DATA_BLEND_FILE=${DATA_BLEND_FILE:-/scratch/project_465002530/users/bmoell/longctx-extend/blend_512k.txt}
exec bash "$DSADIR/lumi/dsa_sparse_roundtrip.sh"
