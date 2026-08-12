#!/bin/bash
#SBATCH --job-name=dsa-cal-512k
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.out
#SBATCH --error=/scratch/project_465002530/users/bmoell/longctx-extend/logs/%j.err
set -euo pipefail

# Evidence-calibrated sparse adaptation probe:
#   * MiniMax MSA: 128-token blocks, 16 blocks, 2,048 selected tokens
#   * LongCat LSA and HiLS: approximately 2K active-token budget
#   * Two 36-update phases give every one of the 36 layers two staggered exact
#     sampled dense-attention-mass recall probes.  Each phase reloads in a fresh
#     process; a final finite update 373 proves the resulting artifact reloads.
export DSADIR=${DSADIR:-$SLURM_SUBMIT_DIR}
export SEQ_LENGTH=524288 CP_SIZE=16 ROTARY_BASE=128000000 RUN_GPU_TESTS=0
export OUT=${DSA_OUT:-/scratch/project_465002530/users/bmoell/longctx-extend/output_dsa_sparse512k_k2048_calibration}
export DATA_BLEND_FILE=${DATA_BLEND_FILE:-/scratch/project_465002530/users/bmoell/superlong_data/mix/data_path.args}
export DATA_CACHE_PATH=${DATA_CACHE_PATH:-/scratch/project_465002530/users/bmoell/longctx-extend/cache_dsa_sparse_524288_superlong}

export DSA_TOPK=2048 DSA_BLOCK_SIZE=128 DSA_ROUTED_BLOCKS=15
export DSA_RECALL_LOG=1 DSA_RECALL_EVERY=36 DSA_RECALL_KS=512,1024,2048
export DSA_GRAD_PROBE=1 DSA_KL_Q_BLOCK=64
export LEARNING_RATE=1e-6 MIN_LEARNING_RATE=1e-6 LR_DECAY_STYLE=constant
export LR_WARMUP_ITERS=0 GLOBAL_BATCH_SIZE=1

exec bash "$DSADIR/lumi/dsa_sparse_calibration.sh"
