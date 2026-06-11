#!/bin/bash
#SBATCH --job-name=megatron-32k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/megatron-32k-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/megatron-32k-%j.err

# 10-step long-context training test on Leonardo (1 node, 4x A100 64GB).
# seq_len=32768, random init, FlashAttention 2, full activation recompute.
#
# Prerequisites:
#   flash-attn installed: sbatch lumi/slurm/install_flash_attn_leonardo.sbatch
#   $WORK/NVIDIA-Megatron-LM, $WORK/megatron_venv, $WORK/data/, $WORK/models/
#
# Memory budget (per GPU, TP=2 PP=2):
#   Model BF16:           ~4.6 GB
#   Optimizer (fp32×3):  ~54.8 GB
#   Activations (full recompute, 32K): ~4 GB
#   Peak total:          ~63 GB  ← just fits in 64 GB A100
#
# Submit: sbatch lumi/slurm/train_32k_test_leonardo.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
MEGATRON_DIR=$WORK/NVIDIA-Megatron-LM
DATA_DIR=$WORK/data
TOKENIZER=$WORK/models/oellm-9b-yarn-v2-32k
OUT_DIR=/leonardo_scratch/large/userexternal/pmoell00/megatron-32k-$SLURM_JOB_ID

mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/tensorboard" "$OUT_DIR/data_cache"

module purge
module load gcc/12.2.0
module load cuda/12.6
module load python/3.11.7

# flash-attn requires GLIBCXX_3.4.29; Python's spack RPATH picks up gcc-8.5.0
# libstdc++ instead of gcc-12.2.0, so we force the right one via LD_PRELOAD.
GCC12_LIB=$(gcc -print-file-name=libstdc++.so.6)
export LD_PRELOAD=$GCC12_LIB${LD_PRELOAD:+:$LD_PRELOAD}

source "$VENV/bin/activate"

# Verify FlashAttention is available
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)" || {
    echo "ERROR: flash-attn not installed. Run install_flash_attn_leonardo.sbatch first."
    exit 1
}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9997
export WORLD_SIZE=$SLURM_NTASKS

export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "=== Megatron 32K test — $(date) ==="
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
echo "World:   $WORLD_SIZE ranks"
echo "Out:     $OUT_DIR"
echo ""

# ── Parallelism: TP=2, PP=2 → 4 GPUs ─────────────────────────────────────────
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=2

# ── Model: 9B architecture with YaRN RoPE at 32K ────────────────────────────
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
    --use-legacy-models
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --no-async-tensor-model-parallel-allreduce
    --use-flash-attn
)

# ── Training ──────────────────────────────────────────────────────────────────
# Full recompute required at 32K: selective recompute stores too much
# activation memory (scales linearly with seq_len).
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4
    --seq-length 32768
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
    --recompute-granularity full
)

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH="1.0 ${DATA_DIR}/multilingual_16k_plus_text_document"

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER"
    --split 900,50,50
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
    --ckpt-format torch
    --tensorboard-dir "$OUT_DIR/tensorboard"
    --log-throughput
)

echo "Launching 32K training..."
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
echo "=== 32K test done — $(date) ==="
