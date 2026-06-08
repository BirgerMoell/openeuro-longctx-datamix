#!/bin/bash
#SBATCH --job-name=megatron-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/megatron-smoke-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/megatron-smoke-%j.err

# 10-step Megatron smoke test on Leonardo (1 node, 4x A100 64GB).
# Uses random init — no checkpoint needed. Validates:
#   - Megatron loads on CUDA (not ROCm)
#   - Data pipeline works with transferred .bin/.idx files
#   - TP=2 PP=2 parallelism works on 4 GPUs
#   - Forward + backward pass completes at seq_len=4096
#
# Prerequisites (run setup_megatron_leonardo.sh first):
#   $WORK/NVIDIA-Megatron-LM   — Megatron fork
#   $WORK/megatron_venv        — Python venv
#   $WORK/data/                — .bin/.idx data files transferred from LUMI
#
# Submit: sbatch lumi/slurm/train_smoke_test_leonardo.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
MEGATRON_DIR=$WORK/NVIDIA-Megatron-LM
DATA_DIR=$WORK/data
TOKENIZER=$WORK/models/oellm-9b-yarn-v2-32k
OUT_DIR=/leonardo_scratch/large/userexternal/pmoell00/megatron-smoke-$SLURM_JOB_ID

mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/tensorboard" "$OUT_DIR/data_cache"

module purge
module load python/3.11.7
module load cuda/12.1

source "$VENV/bin/activate"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS

export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "=== Megatron smoke test — $(date) ==="
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
echo "World:   $WORLD_SIZE ranks"
echo ""

# ── Parallelism: TP=2, PP=2 → 4 GPUs ─────────────────────────────────────────
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=2

# ── Model: 9B architecture ───────────────────────────────────────────────────
MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --ffn-hidden-size 14336
    --max-position-embeddings 32768
    --position-embedding-type rope
    --rotary-base 500000
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --transformer-impl local
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-async-tensor-model-parallel-allreduce
)

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4
    --seq-length 4096
    --train-iters 10
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --recompute-activations
    --recompute-granularity selective
)

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH="1.0 ${DATA_DIR}/multilingual_16k_plus_text_document"

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER"
    --split 990,9,1
    --data-cache-path "$OUT_DIR/data_cache"
    --num-workers 4
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_ARGS=(
    --log-interval 1
    --save-interval 10
    --eval-interval 10
    --eval-iters 2
    --save "$OUT_DIR/checkpoints"
    --tensorboard-dir "$OUT_DIR/tensorboard"
    --log-throughput
)

echo "Launching training..."
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $MEGATRON_DIR/pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${LOG_ARGS[@]}" \
    2>&1

echo ""
echo "=== Smoke test done — $(date) ==="
