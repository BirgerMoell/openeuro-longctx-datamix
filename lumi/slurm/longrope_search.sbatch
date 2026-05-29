#!/bin/bash
#SBATCH --job-name=longrope-search-multilingual
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/longrope-search-multilingual-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longrope-search-multilingual-%j.err

# LongRoPE rescaling-factor search on a multilingual high-resource proxy corpus.
#
# Runs the genetic algorithm (dim_mono) to find per-frequency RoPE scaling
# factors that minimize perplexity at 32K context across 14 European languages.
# Results replace the English-only factors in longrope_multilingual.sbatch.
#
# Parallelism: 4 nodes × 8 GPUs = 32 GPUs, TP=2, 4 evaluators per node
# Target: 32K tokens (same as training target)
# Samples per evaluation: 48 documents
#
# Prerequisites:
#   longrope_search_tokenize.sbatch must have completed first.
#   Dataset must exist at: $DATASET_PATH
#
# After search completes:
#   1. Copy result_final.csv from $RESULT_PATH to a permanent location
#   2. Update longrope_multilingual.sbatch to point --longrope-rescale-factors-path
#      at the new CSV
#   3. After HF conversion, use type "longrope" in config.json (see LUMI_STATUS.md)
#
# Submit: sbatch lumi/slurm/longrope_search.sbatch

set -eox pipefail

WORKDIR=/scratch/project_462000963/bmoell
SEARCH_DIR=/flash/project_462000963/jouni/test/poro2-longrope-search
RESULT_PATH=/flash/project_462000963/bmoell/longrope-search/results/multilingual-proxy

ln -sf "/scratch/project_462000963/bmoell/longrope-search-multilingual-${SLURM_JOBID}.out" \
       "/scratch/project_462000963/bmoell/longrope-search-multilingual-latest.out" 2>/dev/null || true

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc
export CXX=g++

export GA_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export GA_MASTER_PORT=9998

# torchrun rendezvous is node-local
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export PYTHONWARNINGS=ignore

export TARGET_LENGTH=$((32 * 1024))   # 32K — matches training seq_length
TP_SIZE=${TP_SIZE:-2}

# HF model used as the base for search (same tokenizer/weights as training)
export MODEL_PATH=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20

# Multilingual proxy dataset built by longrope_search_tokenize.sbatch
export DATASET_PATH=/flash/project_462000963/bmoell/longrope-search/datasets/multilingual-proxy-tokenized

mkdir -p "$RESULT_PATH"

# Bind mask for LUMI AMD CPUs
c="fe"
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"
export BIND_MASK="$BIND_MASK_2"

export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/var/spool/slurmd
export CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
export launcher="$SEARCH_DIR/launcher.sh"

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""
export PYTHONNOUSERSITE=1

CMD=" \
    evolution/search.py \
    --model $MODEL_PATH \
    --tokenized $DATASET_PATH \
    --algorithm dim_mono \
    --output-dir $RESULT_PATH \
    --target-length $TARGET_LENGTH \
    --dataset-min-tokens $TARGET_LENGTH \
    --attn-implementation flash_attention_2 \
    --truncate \
    --save-memory \
    --master-address $GA_MASTER_ADDR \
    --master-port $GA_MASTER_PORT \
    --nnodes $SLURM_NNODES \
    --evaluators-per-node 4 \
    --tp-size $TP_SIZE \
    --model-ready-timeout 3600 \
    --samples 48 \
"

echo "=== LongRoPE multilingual search — $(date) ==="
echo "Model:   $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output:  $RESULT_PATH"
echo "Nodes:   $SLURM_NNODES"
echo "Target:  ${TARGET_LENGTH} tokens"

srun \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$SEARCH_DIR" \
    -B "$SINGULARITY_BIND" \
    "$CONTAINER" \
    "$launcher" \
    $CMD

echo ""
echo "=== Search complete — $(date) ==="
echo "Result CSV: $RESULT_PATH/result_final.csv"
echo ""
echo "Next steps:"
echo "  1. Update longrope_multilingual.sbatch:"
echo "     --longrope-rescale-factors-path $RESULT_PATH/result_final.csv"
echo "  2. After HF conversion, add to config.json:"
echo "     \"rope_scaling\": {\"type\": \"longrope\", \"factor\": 16.0,"
echo "       \"original_max_position_embeddings\": 2048,"
echo "       \"long_factor\": [...], \"short_factor\": [...]}"
echo "  (see LUMI_STATUS.md for full template)"
