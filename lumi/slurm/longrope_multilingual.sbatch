#!/bin/bash
#SBATCH --job-name=longrope-multilingual
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=32
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longrope-multilingual-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longrope-multilingual-%j.err

# LongRoPE continued pre-training — multilingual 35-language mix.
#
# Extends the OpenEuroLLM 9B model (pre-trained at 2K context) to 32K
# using LongRoPE, fine-tuned on multilingual FinePDFs-Edu data tokenized
# into three length tiers (50% ≥16K, 30% 4–16K, 20% <4K).
#
# Prerequisites:
#   1. tokenize_tiers.sbatch must have completed (produces multilingual bin/idx)
#   2. Jouni's checkpoint at LOAD_CKPT_PATH must be accessible
#
# Parallelism: TP=2, PP=4, CP=4  →  32 nodes × 8 GPUs = 256 GPUs total
#
# Save path: /flash/project_462000963/bmoell/checkpoints/  (NOT Jouni's dir)
#
# Submit: sbatch lumi/slurm/longrope_multilingual.sbatch

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p /scratch/project_462000963/bmoell/logs-multilingual
    sbatch "$0" "$@"
    exit
fi

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
ln -sf "/scratch/project_462000963/bmoell/longrope-multilingual-${SLURM_JOBID}.out" \
       "/scratch/project_462000963/bmoell/longrope-multilingual-latest.out" 2>/dev/null || true

if [[ -v SLURM_RESTART_COUNT ]]; then
    echo "RUN RESTARTED (count: $SLURM_RESTART_COUNT)"
fi

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
export MASTER_PORT=9999
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

OUTPUT_DIR=/flash/project_462000963/bmoell/longrope-multilingual
CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard/${SLURM_JOB_NAME}-${SLURM_JOBID}
DATA_CACHE_PATH=$OUTPUT_DIR/cache_9B_multilingual
mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$DATA_CACHE_PATH"

export HF_HOME="$WORKDIR/hf"

# ── Data — our multilingual three-tier tokenized mix ─────────────────────────
MULTILINGUAL_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual
TOKENIZER_MODEL=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
DATA_PATH="0.5 ${MULTILINGUAL_DIR}/multilingual_16k_plus_text_document \
0.3 ${MULTILINGUAL_DIR}/multilingual_4_16k_text_document \
0.2 ${MULTILINGUAL_DIR}/multilingual_under4k_text_document"

# ── Model (9B: 32 layers, hidden 4096, 32 heads) ──────────────────────────────
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=32
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=32768
ROTARY_BASE=10000

# Load from Jouni's pre-trained checkpoint; save to bmoell's flash dir
LOAD_CKPT_PATH=/flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4
SAVE_CKPT_PATH=$CHECKPOINT_PATH
mkdir -p "$SAVE_CKPT_PATH"

# ── Parallelism (256 GPUs: TP=2, PP=4, CP=4) ─────────────────────────────────
PIPELINE_MODEL_PARALLEL_SIZE=4
TENSOR_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=4
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# ── Optimizer ─────────────────────────────────────────────────────────────────
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=1e-5
MIN_LR=1e-7
WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5
CLIP_GRAD=1.0
WEIGHT_DECAY=0.05

# ── Training ──────────────────────────────────────────────────────────────────
FSDP=0
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
RECOMPUTATION=1
TRAIN_TOKENS=4_194_304_000   # ~1000 iters at GBS=128, SEQ=32768

# ── Output ────────────────────────────────────────────────────────────────────
LOG_INTERVAL=1
SAVE_INTERVAL=500
EVAL_INTERVAL=5000
EVAL_ITERS=100

######################################################################
# DERIVED SETTINGS (do not edit — computed from above)
######################################################################

confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
        echo "Error: variable '$varname' should not be set." >&2
        exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() { echo $((($1+$2-1)/$2)); }

TRAIN_TOKENS=${TRAIN_TOKENS//_}
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

WARMUP_CALC=$((TRAIN_ITERS * ${WARMUP_FRACTION}))
if [ $WARMUP_CALC -gt 25000 ]; then
    LR_WARMUP_ITERS=25000
else
    LR_WARMUP_ITERS=$WARMUP_CALC
fi

LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))
LR_DECAY_ITERS=$TRAIN_ITERS

######################################################################
# BUILD ARGS
######################################################################

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
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
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(--use-torch-fsdp2)
else
    PARALLEL_ARGS=(
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
        --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
        --context-parallel-size $CONTEXT_PARALLEL_SIZE
        --sequence-parallel
        --use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --use-pytorch-profiler
        --profile-ranks 0
        --profile-step-start 5
        --profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type longrope
    --longrope-original-max-position-embeddings 2048
    --longrope-rescale-factors-path /flash/project_462000963/jouni/test/poro2-longrope-search/results/llama3-search2-8B-mn-oellm-9b/result_final.csv
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
)

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(--recompute-activations)
fi

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

if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE)
fi

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
    ${PROFILE_ARGS[@]}
"

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='
echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
