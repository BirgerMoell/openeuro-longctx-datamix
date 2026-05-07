#!/bin/bash
#SBATCH --job-name=yarn-multilingual-test
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/yarn-multilingual-test-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/yarn-multilingual-test-%j.err

# 10-iteration smoke test for the 9B YaRN multilingual training pipeline.
# Uses our own multilingual three-tier tokenized data — the same data and
# weights that yarn_multilingual.sbatch will use for the full run.
#
# Prerequisite: tokenize_tiers.sbatch must have completed and produced:
#   multilingual_16k_plus_text_document.{bin,idx}
#   multilingual_4_16k_text_document.{bin,idx}
#   multilingual_under4k_text_document.{bin,idx}
#
# Parallelism: TP=2, PP=4, CP=4  →  4 nodes × 8 GPUs = 32 GPUs
# CP=1 on 1 node is OOM at seq_length=32768 (60 GB / 64 GB used)
#
# Submit: sbatch lumi/slurm/yarn_multilingual_test.sbatch

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p /scratch/project_462000963/bmoell/logs
    sbatch "$0" "$@"
    exit
fi

ln -sf "/scratch/project_462000963/bmoell/yarn-multilingual-test-${SLURM_JOBID}.out" \
       "/scratch/project_462000963/bmoell/yarn-multilingual-test-latest.out" 2>/dev/null || true

module purge

# ── Environment ──────────────────────────────────────────────────────────────
export CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
export BIND_DIRS=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd

export ROCM_PATH=/opt/rocm-6.4.4
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/usr/lib:${LD_LIBRARY_PATH:-}"
BIND_DIRS="${BIND_DIRS},/opt/rocm/lib/librccl.so:/opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so"

export PYTHONUSERBASE=""
export CC=gcc
export CXX=g++

c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_HMEM=rocr
export FI_HMEM_ROCR_USE_DMABUF=0
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
LAUNCH_SCRIPT=/flash/project_462000963/jouni/oellm/launch.sh

OUTPUT_DIR=$WORKDIR/yarn-multilingual-test-output
CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
DATA_CACHE_PATH=$OUTPUT_DIR/cache
mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH"

export HF_HOME="$WORKDIR/hf"

# ── Data — long-context tier only (avoids BlendedMegatronDataset hang in smoke test)
MULTILINGUAL_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual
TOKENIZER_MODEL=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
DATA_PATH="${MULTILINGUAL_DIR}/multilingual_16k_plus_text_document"

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=32
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=32768
ROTARY_BASE=10000

LOAD_CKPT_PATH=/flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4
SAVE_CKPT_PATH=$CHECKPOINT_PATH
mkdir -p "$SAVE_CKPT_PATH"

# ── Parallelism (1 node = CP=1, TP=2, PP=4 → 8 GPUs) ─────────────────────────
PIPELINE_MODEL_PARALLEL_SIZE=4
TENSOR_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=4

# ── Optimizer ─────────────────────────────────────────────────────────────────
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=1e-5
MIN_LR=1e-7
CLIP_GRAD=1.0
WEIGHT_DECAY=0.05

# ── Training (10 iterations) ──────────────────────────────────────────────────
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
TRAIN_ITERS=10
LR_WARMUP_ITERS=2
LR_DECAY_ITERS=10
LR_WSD_DECAY_ITERS=2
RECOMPUTATION=1

LOG_INTERVAL=1
SAVE_INTERVAL=10000
EVAL_INTERVAL=5
EVAL_ITERS=2

# ── Build args ────────────────────────────────────────────────────────────────
DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 750,150,100
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type single
    --num-workers 1
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type yarn
    --yarn-original-max-position-embeddings 2048
    --yarn-scaling-factor 16.0
    --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --bf16
    --swiglu
    --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --empty-unused-memory-level 2
    --no-load-optim
    --no-load-rng
    --recompute-activations
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --context-parallel-size $CONTEXT_PARALLEL_SIZE
    --sequence-parallel
    --use-distributed-optimizer
    --num-layers-per-virtual-pipeline-stage 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --log-interval $LOG_INTERVAL
)

CHECKPOINT_ARGS=(
    --ckpt-format torch
    --load "$LOAD_CKPT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND="
    $MEGATRON_DIR/pretrain_gpt.py
    ${MODEL_ARGS[@]}
    ${OPTIMIZER_ARGS[@]}
    ${PARALLEL_ARGS[@]}
    ${OUTPUT_ARGS[@]}
    ${CHECKPOINT_ARGS[@]}
    ${DATA_ARGS[@]}
"

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='
echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
