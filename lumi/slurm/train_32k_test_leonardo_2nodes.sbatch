#!/bin/bash
#SBATCH --job-name=megatron-32k-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/megatron-32k-2n-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/megatron-32k-2n-%j.err

# 10-step long-context training test on Leonardo (2 nodes, 8x A100 64GB).
# seq_len=32768, TP=2, PP=2, DP=2, GBS=8, distributed optimizer.
#
# Why 2 nodes: 9B model + Adam optimizer at 32K fills ~60 GB on a single
# A100-64GB, leaving insufficient headroom for the MLP intermediate tensors.
# With DP=2 and --use-distributed-optimizer, Adam states are sharded across
# both DP ranks, saving ~9 GB/GPU and giving ~16 GB of headroom.
#
# Memory budget (per GPU, TP=2 PP=2 DP=2 + distributed optimizer):
#   BF16 model:            ~4.6 GB
#   FP32 grad buffer/2:    ~4.6 GB
#   FP32 master weights/2: ~4.6 GB
#   Adam m1/2:             ~4.6 GB
#   Adam m2/2:             ~4.6 GB
#   BF16 .grad tensors:    ~4.6 GB
#   Checkpoints (full):    ~8 GB
#   NCCL + CUDA:           ~3 GB
#   Peak total:            ~39 GB  ← 24 GB headroom on 64 GB A100
#
# Prerequisites:
#   flash-attn installed: sbatch lumi/slurm/install_flash_attn_leonardo.sbatch
#   $WORK/NVIDIA-Megatron-LM, $WORK/megatron_venv, $WORK/data/, $WORK/models/
#
# Submit: sbatch lumi/slurm/train_32k_test_leonardo_2nodes.sbatch

set -euo pipefail

VENV=$WORK/megatron_venv
MEGATRON_DIR=$WORK/NVIDIA-Megatron-LM
DATA_DIR=$WORK/data
TOKENIZER=$WORK/models/oellm-9b-yarn-v2-32k
OUT_DIR=/leonardo_scratch/large/userexternal/pmoell00/megatron-32k-2n-$SLURM_JOB_ID

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

python -c "import flash_attn; print('flash-attn', flash_attn.__version__)" || {
    echo "ERROR: flash-attn not installed. Run install_flash_attn_leonardo.sbatch first."
    exit 1
}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9997
export WORLD_SIZE=$SLURM_NTASKS   # 8 total (4 per node × 2 nodes)

export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "=== Megatron 32K 2-node test — $(date) ==="
echo "Nodes:   $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
echo "World:   $WORLD_SIZE ranks"
echo "Out:     $OUT_DIR"
echo ""

# ── Parallelism: TP=2, PP=2, DP=2 → 8 GPUs ───────────────────────────────────
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=2
# DP=2 is implicit: WORLD_SIZE / (TP × PP) = 8 / 4 = 2

# ── Model: 9B architecture with YaRN RoPE at 32K ─────────────────────────────
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
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --transformer-impl local
    --use-legacy-models
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --no-async-tensor-model-parallel-allreduce
    --use-flash-attn
)

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
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
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --empty-unused-memory-level 1
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

echo "Launching 32K 2-node training..."
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $MEGATRON_DIR/pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${LOG_ARGS[@]}" \
    2>&1

echo ""
echo "=== 32K 2-node test done — $(date) ==="
